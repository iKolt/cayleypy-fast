"""Fast beam-search engine (T-eng tier): pure-torch, device-agnostic reference.

Implements the plan's key technical insight: for permutation groups with the
dot-product hasher, hashing a neighbor is linear —

    hash(g(s)) = sum_i s[p_g[i]] * v[i] = s . (v o p_g^{-1})

so the hashes of all ``B x G`` neighbors are ONE matmul against precomputed
permuted hash vectors ``VH[j, g] = v[p_g^{-1}[j]]`` — neighbor states are never
materialized. Only the ``beam_width`` top-k survivors are materialized per step
(lazy survivor materialization: one gather of ``B x n``).

``VH`` is derived from the LIVE ``graph.hasher`` instance (``vec_hasher``, or
``vec_hasher_i32`` on the CPU dual-int32 path combined via ``.view(torch.int64)``),
NOT regenerated, so engine hashes bit-match ``hasher.make_hashes`` outputs used
for MITM BFS layers and non-backtracking sets. torch int64/int32 matmul wraps
mod 2^64 / 2^32 exactly like ``StateHasher``, so results are bit-identical
regardless of summation order.

Design constraints baked in (plan, "Constraints discovered in code"):
  * No global sort at target beams: dedup is chunk-wise against a per-step
    ``TorchHashSet``; within a tile, one stable sort + first-occurrence mask.
  * Gen-major flat layout ``(G, T).reshape(-1)`` per tile, so origin math
    (``flat // T``) and dedup-winner ordering match legacy semantics.
  * The legacy ``iterated_batched`` CUDA memory gate is replaced by
    chunk-bounded accounting (no ``MemoryError``; downgrades to per-generator
    selection when the candidate buffer would exceed the budget).
  * Per-chunk op order: dedup vs step accumulator -> MITM check -> ban
    (MITM strictly before ban, cayleypy AGENTS.md section 6.6) -> lazy scoring
    of masked survivors -> selection.
  * Non-backtracking slot bookkeeping matches each legacy mode:
    - "advanced": legacy never resets its history-depth slots, so the ban is
      effectively a single ever-growing seen set (cumulative ban);
    - "iterated" / "iterated_batched": current slot reset per step (sliding
      window of the last ``history_depth`` steps).
    Both add the chunk's post-dedup PRE-ban hashes to the history, then apply
    the ban mask (legacy ordering).

Replicated quirk (do not fix here): the hamming predictor always measures
distance to ``graph.central_state`` even when ``destination_state`` is custom
(cayleypy ``predictor.py``, via ``_init_predictor`` default). The engine
reproduces this byte-for-byte for parity.
"""

import contextlib
import time
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

# pylint: disable=protected-access
# By design: cayleypy-fast is an in-place accelerator for cayleypy's alpha
# internals; the probe (_probe.py) verifies every protected symbol imported here.
from cayleypy.algo.beam_search import (
    BeamSearchResult,
    _BeamSearchProfile,
    _check_path_found,
    _cuda_sync,
    _early_return_if_at_dest,
    _encode_and_dedupe_start,
    _finalize_not_found,
    _init_predictor,
    _precompute_mitm,
    _restore_path,
    _setup_path_device_and_restore,
)
from cayleypy.predictor import Predictor
from cayleypy.torch_utils import TorchHashSet, isin_via_searchsorted

if TYPE_CHECKING:
    from cayleypy import CayleyGraph
    from cayleypy.bfs_result import BfsResult


__all__ = [
    "FastBeamEngine",
    "PermutedHashVectors",
    "build_permuted_hash_vectors",
    "create_engine",
    "engine_available",
    "hash_neighbors_mulsum",
    "hash_neighbors_tiled",
    "resolve_backend",
]

_MODES = ("simple", "advanced", "iterated", "iterated_batched")

# Byte budget for one survivor-materialization chunk: ``_materialize`` gathers the
# int64 permutation-index tensor ``perms[gens]`` of shape (K, state_size) — at
# bw=2^24 on cube333 this single tensor alone was 6.75 GiB and OOM'd a 16 GB
# P100 (Kaggle gpu-perf, cube333_iterated_bw2p24). Chunking bounds it.
_MATERIALIZE_CHUNK_BYTES = 2**27

# Ban bookkeeping varies by mode (see module docstring).
_BAN_BY_MODE = {"simple": "none", "advanced": "cumulative", "iterated": "window", "iterated_batched": "window"}


@contextlib.contextmanager
def _timed(profile: Optional[_BeamSearchProfile], region: str):
    """Time a profile region with GPU-synced brackets (legacy verbose>=100 semantics)."""
    if profile is None:
        yield
        return
    _cuda_sync()
    t1 = time.time()
    try:
        yield
    finally:
        _cuda_sync()
        setattr(profile, region, getattr(profile, region) + (time.time() - t1))


# -----------------------------------------------------------------------------
# Permuted hash vectors (plan, "Key technical insight").
# -----------------------------------------------------------------------------


class PermutedHashVectors:
    """Per-generator permuted hash vectors derived from the live ``graph.hasher``.

    ``vh`` has shape ``(state_size, n_generators)`` int64 for the int64 hasher
    path, or ``(state_size, 2 * n_generators)`` int32 for the CPU dual-int32
    path (two int32 hashes per generator, combined via ``.view(torch.int64)``).

    ``backend`` selects the int64 (CUDA) hash rung: ``"triton" | "matmul" |
    "mulsum"``; ``None`` means "default" (torch matmul) and is also the value
    for dual-int32 CPU instances, whose numba/torch dispatch is backend-
    independent. All int64 rungs are bit-equal by construction (int64 mul/add
    wrap mod 2^64), so mid-search demotion cannot change search semantics.
    """

    __slots__ = ("vh", "dual_int32", "_numba_vh_t", "backend", "_triton_vh_t")

    def __init__(self, vh: torch.Tensor, dual_int32: bool) -> None:
        self.vh = vh
        self.dual_int32 = dual_int32
        # Lazily built (n, 2*G)-transposed numpy copy for the optional numba tier.
        self._numba_vh_t: Optional[np.ndarray] = None
        # int64 hash backend (see class docstring); resolved once per engine.
        self.backend: Optional[str] = None
        # Lazily built (G, n)-transposed contiguous CUDA tensor for the Triton tier.
        self._triton_vh_t: Optional[torch.Tensor] = None


def _permute_vector(v: torch.Tensor, inv_perms: torch.Tensor, dual_int32: bool) -> torch.Tensor:
    """Compute ``VH[j, g] = v[p_g^{-1}[j]]`` from hash vector(s) ``v`` and inverse permutations.

    :param v: ``(n,)`` int64 (single-hash path) or ``(n, 2)`` int32 (dual-int32 path).
    :param inv_perms: ``(n_generators, n)`` int64; ``inv_perms[g] = p_g^{-1}``.
    :param dual_int32: True for the CPU dual-int32 hasher path.
    :return: ``(n, n_generators)`` int64, or ``(n, 2 * n_generators)`` int32 when ``dual_int32``.
    """
    n_generators = inv_perms.shape[0]
    n = inv_perms.shape[1]
    if dual_int32:
        # v[inv] is (G, n, 2); want (n, G*2) with VH32[j, 2g:2g+2] = v[inv[g, j]].
        return v[inv_perms.long()].permute(1, 0, 2).reshape(n, 2 * n_generators).contiguous()
    return v[inv_perms.long()].t().contiguous()


def build_permuted_hash_vectors(graph: "CayleyGraph") -> Optional[PermutedHashVectors]:
    """Derive permuted hash vectors from the LIVE ``graph.hasher`` instance.

    Returns ``None`` when the dot-product fast path is unavailable: non-permutation
    (matrix) groups, bit-encoded states (splitmix64 hasher), or the identity
    hasher (``encoded_state_size == 1``).
    """
    if not graph.definition.is_permutation_group():
        return None
    if graph.string_encoder is not None:
        return None
    hasher = graph.hasher
    if hasher.is_identity:
        return None
    inv_perms = torch.argsort(graph.permutations_torch, dim=1)
    if hasattr(hasher, "vec_hasher_i32"):
        # CPU dual-int32 path.
        return PermutedHashVectors(_permute_vector(hasher.vec_hasher_i32, inv_perms, dual_int32=True), dual_int32=True)
    if hasattr(hasher, "vec_hasher"):
        # int64 path (GPU; also the "older GPU" mul+sum variant, which stores the
        # same vector reshaped to (n,) — flatten covers both (n, 1) and (n,)).
        return PermutedHashVectors(_permute_vector(hasher.vec_hasher.reshape(-1), inv_perms, dual_int32=False), False)
    return None


# -----------------------------------------------------------------------------
# int64 (CUDA) hash backend ladder (plan, "hash backend ladder").
# -----------------------------------------------------------------------------

# Demotion order for mid-search failures: triton -> matmul -> mulsum (floor).
_DEMOTE_NEXT = {"triton": "matmul", "matmul": "mulsum"}


def hash_neighbors_mulsum(hv: PermutedHashVectors, states: torch.Tensor) -> torch.Tensor:
    """Rung-3 int64 fallback: per-generator multiply+sum (no integer GEMM needed).

    Mirrors the hasher's "older GPU" path (``sum(states * v)``) one generator at
    a time; the per-generator temp is bounded by ``B x n`` int64 (B is already
    capped by ``FastBeamEngine._HASH_SUBTILE_ROWS``). Bit-equal to the matmul
    rung: int64 mul/add wrap mod 2^64 regardless of summation order.
    """
    assert not hv.dual_int32, "mulsum fallback is for the int64 path only"
    s64 = states.to(torch.int64)
    n_generators = hv.vh.shape[1]
    out = torch.empty((states.shape[0], n_generators), dtype=torch.int64, device=s64.device)
    for g in range(n_generators):
        out[:, g] = (s64 * hv.vh[:, g]).sum(dim=1)
    return out


def _hash_via_matmul(hv: PermutedHashVectors, states: torch.Tensor) -> torch.Tensor:
    """Rung-2 int64 path: one torch matmul. Wraps mod 2^64; needs integer GEMM support."""
    return states.to(torch.int64) @ hv.vh  # (B, G) int64


def _hash_via_triton(hv: PermutedHashVectors, states: torch.Tensor) -> torch.Tensor:
    """Rung-1 int64 path: the Triton kernel (see triton_kernels.py)."""
    from .triton_kernels import hash_neighbors_int64_triton  # pylint: disable=import-outside-toplevel

    return hash_neighbors_int64_triton(hv, states)


def _hash_int64_dispatch(hv: PermutedHashVectors, states: torch.Tensor) -> torch.Tensor:
    """Dispatch the int64 hash via ``hv.backend`` with mid-search demotion.

    A runtime failure on the triton or matmul rung warns once (the backend is
    mutated, so it cannot fail twice), demotes to the next rung, and recomputes
    the failed subtile — hashing is a pure function of the inputs and all rungs
    are bit-equal, so demotion never changes search semantics.
    """
    while True:
        backend = hv.backend
        try:
            if backend == "triton":
                return _hash_via_triton(hv, states)
            if backend == "mulsum":
                return hash_neighbors_mulsum(hv, states)
            return _hash_via_matmul(hv, states)  # None (unresolved/CPU) or "matmul".
        # Deliberate broad catch: demote and recompute, never abort a running search.
        # pylint: disable=broad-exception-caught
        except Exception as exc:
            nxt = _DEMOTE_NEXT.get(backend if backend is not None else "matmul")
            if nxt is None:
                raise
            warnings.warn(
                f"cayleypy-fast: hash backend '{backend or 'matmul'}' failed mid-search "
                f"({type(exc).__name__}: {exc}); demoting to '{nxt}'.",
                RuntimeWarning,
                stacklevel=2,
            )
            hv.backend = nxt


def resolve_backend(hv: PermutedHashVectors) -> None:
    """Resolve ``hv.backend`` once per engine (int64 CUDA instances only).

    Ladder: triton (kernel probe-compiled and bit-checked against the mul+sum
    reference) -> matmul (mirrors the hasher's own try/except, hasher.py) ->
    mulsum (hasher-mirrored floor, bit-equal by construction). No-op for the
    CPU dual-int32 path (numba/torch dispatch there is untouched) and for CPU
    int64 instances (matmul works there; ``None`` keeps the historical path).
    """
    if hv.dual_int32 or not hv.vh.is_cuda:
        hv.backend = None
        return
    from .triton_kernels import triton_available  # pylint: disable=import-outside-toplevel

    device = hv.vh.device
    n = hv.vh.shape[0]
    # Deterministic probe inputs (values < n, like real permutation states), in
    # both dtypes the kernel casts from. No global RNG consumed.
    probes = []
    base = torch.arange(8 * n, device=device, dtype=torch.int64).reshape(8, n) % max(n, 1)
    probes.append(base)
    if n <= 127:
        probes.append(base.to(torch.int8))
    reference = [hash_neighbors_mulsum(hv, p) for p in probes]
    if triton_available():
        try:
            for p, ref in zip(probes, reference):
                if not torch.equal(_hash_via_triton(hv, p), ref):
                    raise RuntimeError("triton probe bit-mismatch vs mul+sum reference")
            hv.backend = "triton"
            return
        # Probe failure falls through to the next rung (any exception type).
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    try:
        for p, ref in zip(probes, reference):
            if not torch.equal(_hash_via_matmul(hv, p), ref):
                raise RuntimeError("matmul probe bit-mismatch vs mul+sum reference")
        hv.backend = "matmul"
        return
    # e.g. no integer GEMM on this GPU (hasher.py try/excepts likewise).
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    hv.backend = "mulsum"


def hash_neighbors_tiled(hv: PermutedHashVectors, states: torch.Tensor) -> torch.Tensor:
    """Hash all ``B x G`` neighbors of ``states`` without materializing them.

    :param hv: Permuted hash vectors from :func:`build_permuted_hash_vectors`.
    :param states: ``(B, state_size)`` encoded states (any integer dtype).
    :return: int64 hashes of shape ``(B, n_generators)`` — row = beam state,
        column = generator. Bit-identical to ``graph.hasher.make_hashes`` run on
        the materialized neighbor states.

    Dispatch: the optional numba tier (T4) handles the CPU dual-int32 case
    (13-39x faster than torch's non-BLAS int32 matmul; bit-equal), torch
    otherwise. ``CAYLEYPY_FAST_NUMBA_DISABLE=1`` forces torch. The int64
    (CUDA) case dispatches through the ``hv.backend`` ladder resolved by
    :func:`resolve_backend` (triton -> matmul -> mulsum, mid-search demotion
    on runtime failure).
    """
    if hv.dual_int32:
        try:
            from .numba_kernels import (  # pylint: disable=import-outside-toplevel
                hash_neighbors_dual_int32_numba,
                numba_available,
            )

            if numba_available():
                return hash_neighbors_dual_int32_numba(hv, states)
        except ImportError:
            pass  # numba tier not installed; torch path below.
        h32 = states.to(torch.int32) @ hv.vh  # (B, 2G) int32; wraps mod 2^32 per element.
        n_generators = hv.vh.shape[1] // 2
        return h32.view(-1, n_generators, 2).view(torch.int64).reshape(-1, n_generators)
    return _hash_int64_dispatch(hv, states)


def engine_available(graph: "CayleyGraph") -> bool:
    """Fast-path availability gate: permutation group + plain states + dot-product hasher."""
    return build_permuted_hash_vectors(graph) is not None


# -----------------------------------------------------------------------------
# Selection policies (plan, "step()" item 6).
# -----------------------------------------------------------------------------


class _GlobalTopK:
    """Running global top-k selection (simple/advanced).

    Scores are computed lazily: while the total candidate count stays within the
    beam width, candidates are buffered raw (mirrors the legacy "not scored cause
    beam_width is big enough" branch). Once the beam overflows, everything is
    scored and a running top-k buffer of size <= k is maintained (each merge
    touches at most 2k elements).
    """

    def __init__(self, beam_width: int) -> None:
        self.k = beam_width
        self.total = 0
        self.scored = False
        self.raw_hashes: list[torch.Tensor] = []
        self.raw_parents: list[torch.Tensor] = []
        self.raw_gens: list[torch.Tensor] = []
        self.scores: Optional[torch.Tensor] = None
        self.hashes: Optional[torch.Tensor] = None
        self.parents: Optional[torch.Tensor] = None
        self.gens: Optional[torch.Tensor] = None

    def _score_buffered(self, score_fn) -> None:
        self.hashes = torch.cat(self.raw_hashes)
        self.parents = torch.cat(self.raw_parents)
        self.gens = torch.cat(self.raw_gens)
        self.scores = score_fn(self.parents, self.gens)
        self.raw_hashes, self.raw_parents, self.raw_gens = [], [], []
        self.scored = True

    def offer(self, hashes: torch.Tensor, parents: torch.Tensor, gens: torch.Tensor, score_fn) -> None:
        if hashes.numel() == 0:
            return
        self.total += hashes.numel()
        if not self.scored and self.total <= self.k:
            self.raw_hashes.append(hashes)
            self.raw_parents.append(parents)
            self.raw_gens.append(gens)
            return
        if not self.scored:
            if self.raw_hashes:
                self._score_buffered(score_fn)
            else:
                # First offered chunk alone exceeds the beam: nothing buffered.
                scores = score_fn(parents, gens)
                if scores.numel() > self.k:
                    # Cap immediately: the typical engine step has exactly ONE
                    # offer (whole beam fits one tile), so without this top-k the
                    # beam would grow geometrically step over step.
                    keep = torch.topk(scores, k=self.k, largest=False, sorted=False).indices
                    scores, hashes, parents, gens = scores[keep], hashes[keep], parents[keep], gens[keep]
                self.scores = scores
                self.hashes, self.parents, self.gens = hashes, parents, gens
                self.scored = True
                return
        scores = score_fn(parents, gens)
        if scores.numel() > self.k:
            # Pre-shrink the chunk so each merge touches at most 2k elements.
            keep = torch.topk(scores, k=self.k, largest=False, sorted=False).indices
            scores, hashes, parents, gens = scores[keep], hashes[keep], parents[keep], gens[keep]
        assert (
            self.scores is not None and self.hashes is not None and self.parents is not None and self.gens is not None
        )
        all_scores = torch.cat([self.scores, scores])
        keep = torch.topk(all_scores, k=min(self.k, all_scores.numel()), largest=False, sorted=False).indices
        self.scores = all_scores[keep]
        self.hashes = torch.cat([self.hashes, hashes])[keep]
        self.parents = torch.cat([self.parents, parents])[keep]
        self.gens = torch.cat([self.gens, gens])[keep]

    def finalize(self, _score_fn) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Any]:
        """Return (hashes, parents, gens, best_score|None)."""
        if not self.scored:
            if not self.raw_hashes:
                return None, None, None, None
            return torch.cat(self.raw_hashes), torch.cat(self.raw_parents), torch.cat(self.raw_gens), None
        assert self.scores is not None
        assert self.hashes is not None and self.hashes.numel() <= self.k
        best = float(self.scores.min()) if self.scores.numel() > 0 else None
        return self.hashes, self.parents, self.gens, best


class _PerGenTopK:
    """Per-generator running top-k (iterated): each generator keeps up to
    ``beam_width // n_generators`` survivors, preserving per-generator fairness."""

    def __init__(self, n_generators: int, part: int) -> None:
        self.n_generators = n_generators
        self.slots = [_GlobalTopK(part) for _ in range(n_generators)]

    def offer(self, hashes: torch.Tensor, parents: torch.Tensor, gens: torch.Tensor, score_fn) -> None:
        for g in range(self.n_generators):
            idx = torch.where(gens == g)[0]
            self.slots[g].offer(hashes[idx], parents[idx], gens[idx], score_fn)

    def finalize(self, score_fn) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Any]:
        hs, ps, gs, best = [], [], [], None
        for slot in self.slots:
            fh, fp, fg, per_gen_best = slot.finalize(score_fn)
            if fh is None:
                continue
            assert fp is not None and fg is not None
            hs.append(fh)
            ps.append(fp)
            gs.append(fg)
            # Legacy debug formula (iterated): min over scored chunks of the selected-min.
            if per_gen_best is not None:
                best = per_gen_best if best is None else min(best, per_gen_best)
        if not hs:
            return None, None, None, None
        return torch.cat(hs), torch.cat(ps), torch.cat(gs), best


class _BatchedRedistribute:
    """iterated_batched selection: buffer all candidates, then one scoring pass
    with per-generator top-k + legacy surplus redistribution (global pool fill)."""

    def __init__(self, beam_width: int, part: int, n_generators: int) -> None:
        self.k = beam_width
        self.part = part
        self.n_generators = n_generators
        self.total = 0
        self.hashes: list[torch.Tensor] = []
        self.parents: list[torch.Tensor] = []
        self.gens: list[torch.Tensor] = []

    def offer(self, hashes: torch.Tensor, parents: torch.Tensor, gens: torch.Tensor, _score_fn) -> None:
        self.total += hashes.numel()
        self.hashes.append(hashes)
        self.parents.append(parents)
        self.gens.append(gens)

    def finalize(self, score_fn) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Any]:
        if self.total == 0:
            return None, None, None, None
        hashes = torch.cat(self.hashes)
        parents = torch.cat(self.parents)
        gens = torch.cat(self.gens)
        if self.total <= self.k:
            # Legacy "not scored cause beam_width is big enough" branch.
            return hashes, parents, gens, None
        # Score ALL survivors once (one logical predictor pass, streamed in sub-batches).
        scores = score_fn(parents, gens)

        # Per-generator selection (up to `part` each), mapping topk indices back
        # to GLOBAL indices (legacy "Task 0" surplus fix).
        sel_idx_list = []
        used = 0
        for g in range(self.n_generators):
            g_idx = torch.where(gens == g)[0]
            if g_idx.numel() > self.part:
                top = torch.topk(scores[g_idx], k=self.part, largest=False, sorted=True).indices
                g_idx = g_idx[top]
            sel_idx_list.append(g_idx)
            used += g_idx.numel()

        # Surplus redistribution: if total selected < beam_width, fill remaining
        # slots from the global pool (survivors not yet selected), by score.
        if used < self.k:
            selected_mask = torch.zeros(self.total, dtype=torch.bool, device=hashes.device)
            for g_idx in sel_idx_list:
                selected_mask[g_idx] = True
            remaining = torch.where(~selected_mask)[0]
            n_fill = min(self.k - used, remaining.numel())
            if n_fill > 0:
                top = torch.topk(scores[remaining], k=n_fill, largest=False, sorted=True).indices
                sel_idx_list.append(remaining[top])

        sel = torch.cat(sel_idx_list)[: self.k]
        # Legacy debug formula (iterated_batched): global min of ALL survivor scores.
        return hashes[sel], parents[sel], gens[sel], float(scores.min())


# -----------------------------------------------------------------------------
# Engine.
# -----------------------------------------------------------------------------


class FastBeamEngine:
    """Chunked beam-search engine for one graph (plan, section "Engine").

    Step structure: two-level chunking (state tile x generator), gen-major
    within a tile:

      1. hash chunk via matmul with ``VH`` (CPU: dual-int32 pair -> view int64);
      2. dedup vs the per-step ``TorchHashSet`` (sorted add);
      3. MITM check (``isin_via_searchsorted`` vs BFS layer hashes) — early exit
         keeps the ``i_step + bfs_layer_id`` arithmetic;
      4. non-backtracking ban (per-mode bookkeeping, legacy ordering);
      5. lazy scoring of masked survivors (hamming fused; hybrid streams the
         decoded survivor states through the user predictor);
      6. selection per mode (running global top-k / per-generator top-k /
         batched with surplus redistribution);
      7. materialize survivors with one gather; record ``restore_path_hashes``.
    """

    # Subtile rows for the state-cast + matmul (bounds the cast temp: for
    # cube555, 2^18 * 150 * 8B = 315 MB int64 / 157 MB int32).
    _HASH_SUBTILE_ROWS = 2**18
    # Batch size for streaming hybrid (NN/custom) predictor calls.
    _SCORE_BATCH = 2**18

    def __init__(self, graph: "CayleyGraph", hv: PermutedHashVectors) -> None:
        self.graph = graph
        self.hv = hv
        self.perms = graph.permutations_torch  # (G, n) int64
        self.inv_perms = torch.argsort(self.perms, dim=1)
        self.n_generators = graph.definition.n_generators
        central_enc = graph.encode_states(graph.central_state).reshape(-1)
        # Per-generator permuted central state for fused hamming scoring:
        # hamming(g(s), c) = hamming(s, c o p_g^{-1}), so survivor states are
        # never materialized for scoring. Replicates the legacy quirk of always
        # measuring distance to central_state (TODO(char-spec) upstream).
        self.central_by_gen = central_enc[self.inv_perms].contiguous()  # (G, n), graph dtype
        # Resolve the int64 (CUDA) hash backend ladder once per engine (engines
        # are cached per graph, so probes/JIT compile never hit the hot path).
        resolve_backend(hv)
        self.tile_states = self._choose_tile_states()

    # -- Precompute ------------------------------------------------------

    def _choose_tile_states(self) -> int:
        """Tile size from the memory limit (plan, "Memory model").

        Per-tile temps: ~24 bytes per candidate (int64 hash + int64 sort index +
        sort workspace), tile hashes only — candidate states are never
        materialized. Budget: 1/8 of ``memory_limit`` for tile temps.
        """
        budget_bytes = self.graph.memory_limit_bytes // 8
        per_state_row = 24 * self.n_generators
        tile = budget_bytes // max(per_state_row, 1)
        return int(min(2**22, max(2**12, tile)))

    # -- Hashing ---------------------------------------------------------

    def _hash_tile_flat(self, tile_states: torch.Tensor) -> torch.Tensor:
        """Hash all G neighbors of each tile state; return ``(G*T,)`` int64 gen-major flat."""
        outs = []
        for z in tile_states.split(self._HASH_SUBTILE_ROWS):
            h64 = hash_neighbors_tiled(self.hv, z)  # (t, G); numba tier dispatches inside.
            outs.append(h64)
        hashes = outs[0] if len(outs) == 1 else torch.cat(outs)
        return hashes.t().reshape(-1)  # Gen-major flat: index = g * T + t.

    def _tile_candidates(self, tile_states, tile_start, step_dedup, profile):
        """Per-tile steps 1-2: hash, stable sort, within-tile dedup, dedup vs the step set.

        Returns (hashes_sorted, parents, gens) of the surviving candidates.
        """
        tile_rows = tile_states.shape[0]
        with _timed(profile, "hash"):
            flat = self._hash_tile_flat(tile_states)
        with _timed(profile, "sort"):
            flat_sorted, sort_idx = torch.sort(flat, stable=True)
            dedup_mask = torch.ones(flat_sorted.shape[0], dtype=torch.bool, device=flat.device)
            if flat_sorted.shape[0] > 1:
                dedup_mask[1:] = flat_sorted[1:] != flat_sorted[:-1]
            unique_flat = sort_idx[dedup_mask]
            hashes = flat_sorted[dedup_mask]  # Sorted (preserves cayleypy AGENTS.md section 6 invariant).
            parents = unique_flat % tile_rows + tile_start
            gens = unique_flat // tile_rows
        with _timed(profile, "dedup"):
            if len(step_dedup) > 0:
                keep = step_dedup.get_mask_to_remove_seen_hashes(hashes)
                hashes, parents, gens = hashes[keep], parents[keep], gens[keep]
            if hashes.numel() > 0:
                step_dedup.add_sorted_hashes(hashes)
        return hashes, parents, gens

    # -- Non-backtracking ban (step 4) -----------------------------------

    @staticmethod
    def _apply_cumulative_ban(seen_set, hashes, parents, gens, profile):
        # "advanced" legacy semantics: one ever-growing seen set. Query before
        # adding this chunk's post-dedup pre-ban hashes (legacy ordering).
        with _timed(profile, "isin"):
            mask = seen_set.get_mask_to_remove_seen_hashes(hashes)
            seen_set.add_sorted_hashes(hashes)
            return hashes[mask], parents[mask], gens[mask]

    @staticmethod
    def _apply_window_ban(slots, i_current, hashes, parents, gens, profile):
        # "iterated"/"iterated_batched" legacy semantics: sliding window of the
        # last history_depth steps; current slot was reset at step start.
        with _timed(profile, "isin"):
            mask = torch.ones_like(hashes, dtype=torch.bool)
            for slot in slots:
                mask &= slot.get_mask_to_remove_seen_hashes(hashes)
            slots[i_current].add_sorted_hashes(hashes)  # Pre-ban add (legacy ordering).
            return hashes[mask], parents[mask], gens[mask]

    # -- Scoring (step 5) -------------------------------------------------

    def _make_score_fn(self, beam_states, hamming: bool, predictor: Optional[Predictor]):
        """Build the lazy scoring closure for this step.

        Hamming is fused: score(parent, g) = hamming(beam[parent], central o p_g^{-1}),
        no neighbor materialization. Hybrid streams decoded survivor states
        through the user predictor in ``_SCORE_BATCH`` batches.
        """
        graph = self.graph
        if hamming:
            central_by_gen = self.central_by_gen

            def hamming_scores(parents: torch.Tensor, gens: torch.Tensor) -> torch.Tensor:
                rows = beam_states[parents]
                if rows.dim() == 1:
                    rows = rows.unsqueeze(0)
                return (rows != central_by_gen[gens]).sum(dim=1, dtype=torch.int32)

            return hamming_scores

        assert predictor is not None
        perms = self.perms
        score_batch = self._SCORE_BATCH

        def predictor_scores(parents: torch.Tensor, gens: torch.Tensor) -> torch.Tensor:
            outs = []
            for start in range(0, parents.numel(), score_batch):
                chunk_parents = parents[start : start + score_batch]
                chunk_gens = gens[start : start + score_batch]
                rows = beam_states[chunk_parents]
                if rows.dim() == 1:
                    rows = rows.unsqueeze(0)
                neighbors = torch.gather(rows, 1, perms[chunk_gens])
                scores = predictor(graph.decode_states(neighbors))
                if not isinstance(scores, torch.Tensor):
                    scores = torch.as_tensor(np.asarray(scores))
                outs.append(scores.to(graph.device).reshape(-1))
            return outs[0] if len(outs) == 1 else torch.cat(outs)

        return predictor_scores

    # -- Survivor materialization (lazy; step 7 + MITM middle state) ------

    def _materialize(self, beam_states, parents, gens):
        """Build neighbor states for the given (parent, gen) pairs with one gather.

        Chunked: the int64 gather index ``perms[gens]`` is (K, state_size) — at
        bw=2^24, n=54 that is 6.75 GiB (Kaggle P100 OOM). Row-wise chunking
        keeps the index temp under ``_MATERIALIZE_CHUNK_BYTES``; results are
        bit-identical (row-wise op, order-preserving concat).
        """
        total = parents.numel()
        chunk = max(1, _MATERIALIZE_CHUNK_BYTES // (self.perms.shape[1] * 8))
        if total > chunk:
            outs = []
            for s in range(0, total, chunk):
                rows_c = beam_states[parents[s : s + chunk]]
                if rows_c.dim() == 1:
                    rows_c = rows_c.unsqueeze(0)
                outs.append(torch.gather(rows_c, 1, self.perms[gens[s : s + chunk]]))
            return outs[0] if len(outs) == 1 else torch.cat(outs)
        rows = beam_states[parents]
        if rows.dim() == 1:
            rows = rows.unsqueeze(0)
        return torch.gather(rows, 1, self.perms[gens])

    def _matching_states(self, beam_states, hashes, parents, gens, layer_hashes):
        """Materialize only the candidates whose hash is in the MITM layer found."""
        # isin_via_searchsorted(elements, test_elements_sorted): the mask is over
        # the candidates (elements), the BFS layer is the sorted test set.
        mask = isin_via_searchsorted(hashes, layer_hashes.to(self.graph.device))
        rows = torch.where(mask)[0]
        return hashes[rows], self._materialize(beam_states, parents[rows], gens[rows])

    # -- Main loop --------------------------------------------------------

    # pylint: disable=too-many-locals
    def _run(
        self,
        mode: str,
        start_state: Any,
        destination_state: Any,
        predictor: Any,
        beam_width: int,
        max_steps: int,
        return_path: bool,
        path_device: Any,
        history_depth: int,
        hashed_neigbourhood: Any,
        memory_cleanup: bool,
        verbose: int,
    ) -> BeamSearchResult:
        assert mode in _MODES
        graph = self.graph
        debug_scores: dict[int, float] = {}
        n_generators = self.n_generators

        if mode in ("iterated", "iterated_batched"):
            # Legacy validation (B2), replicated because the legacy method never runs.
            if beam_width < n_generators:
                which = "iterated" if mode == "iterated" else "iterated batched"
                detail = (
                    "otherwise iterated modes cannot allocate per-generator slots."
                    if mode == "iterated"
                    else "otherwise per-generator slots cannot be allocated."
                )
                raise ValueError(
                    f"beam_width ({beam_width}) must be >= n_generators ({n_generators}) for {which} beam search, "
                    f"{detail}"
                )

        # Hamming detection uses the RAW predictor argument, before _init_predictor
        # wrapping (plan, "Hybrid predictors").
        hamming = predictor is None or (isinstance(predictor, str) and predictor == "hamming")
        _predictor = None if hamming else _init_predictor(graph, predictor)

        # Shared preamble helpers (probe-verified cayleypy internals).
        if destination_state is None:
            destination_state = graph.central_state
        beam_states, beam_hashes, _dest_hashes = _encode_and_dedupe_start(graph, start_state, destination_state)
        path_device, restore_path_hashes = _setup_path_device_and_restore(path_device, return_path, beam_hashes, graph)
        early_result = _early_return_if_at_dest(beam_hashes, _dest_hashes, debug_scores, graph)
        if early_result is not None:
            return early_result
        bfs_result_for_mitm, bfs_layers_hashes = _precompute_mitm(
            graph, hashed_neigbourhood, destination_state, path_device
        )

        # Non-backtracking bookkeeping (per-mode ban semantics; see module docstring).
        # Both containers are always created (cheap when empty) so their use in
        # the loop is definite; the active one is selected by `ban_mode`.
        ban_mode = _BAN_BY_MODE[mode] if history_depth > 0 else "none"
        nonbacktrack_slots = [TorchHashSet() for _ in range(history_depth)] if ban_mode == "window" else []
        seen_set = TorchHashSet()
        i_cyclic_index = 0

        # Selection policy: iterated_batched falls back to per-generator slots when
        # the buffered candidate metadata would exceed the chunk-bounded budget
        # (engine replacement for the legacy CUDA memory gate; ~32 B/candidate).
        beam_width_part = beam_width // n_generators
        use_batched = mode == "iterated_batched"
        if use_batched and beam_width * n_generators * 32 > graph.memory_limit_bytes // 4:
            if verbose >= 1:
                est_gb = beam_width * n_generators * 32 / 2**30
                print(
                    f"cayleypy-fast: candidate buffer ~{est_gb:.1f} GB exceeds budget; using per-generator selection."
                )
            use_batched = False

        verbose_step = mode != "simple"  # Message wording differs per mode (legacy verbatim).

        t0 = time.time()
        profile = _BeamSearchProfile() if verbose >= 100 else None
        for i_step in range(1, max_steps + 1):
            if profile is not None:
                _cuda_sync()
                profile.reset_step()

            if ban_mode == "window":
                # Reset the current slot (legacy Task 1.6) before adding this step's hashes.
                i_cyclic_index = (i_cyclic_index + 1) % history_depth
                nonbacktrack_slots[i_cyclic_index].data = []

            step_dedup = TorchHashSet()
            if use_batched:
                policy: Any = _BatchedRedistribute(beam_width, beam_width_part, n_generators)
            elif mode == "iterated":
                policy = _PerGenTopK(n_generators, beam_width_part)
            else:  # simple / advanced
                policy = _GlobalTopK(beam_width)

            score_fn = self._make_score_fn(beam_states, hamming, _predictor)
            any_preban_candidates = False

            beam_rows = beam_states.shape[0]
            for tile_start in range(0, beam_rows, self.tile_states):
                tile = beam_states[tile_start : tile_start + self.tile_states]
                hashes, parents, gens = self._tile_candidates(tile, tile_start, step_dedup, profile)
                if hashes.numel() == 0:
                    continue
                any_preban_candidates = True

                # Step 3: MITM check (strictly before the ban, cayleypy section 6.6).
                with _timed(profile, "check"):
                    bfs_layer_id = _check_path_found(hashes.to(path_device), bfs_layers_hashes)
                if bfs_layer_id != -1:
                    path = None
                    if return_path:
                        mid_hashes, mid_states = self._matching_states(
                            beam_states, hashes, parents, gens, bfs_layers_hashes[bfs_layer_id]
                        )
                        path = _restore_path(
                            bfs_layer_id,
                            mid_hashes,
                            mid_states,
                            graph,
                            restore_path_hashes,
                            bfs_layers_hashes,
                            bfs_result_for_mitm,
                            destination_state,
                        )
                    return BeamSearchResult(True, i_step + bfs_layer_id, path, debug_scores, graph.definition)

                # Step 4: non-backtracking ban.
                if ban_mode == "cumulative":
                    hashes, parents, gens = self._apply_cumulative_ban(seen_set, hashes, parents, gens, profile)
                elif ban_mode == "window":
                    hashes, parents, gens = self._apply_window_ban(
                        nonbacktrack_slots, i_cyclic_index, hashes, parents, gens, profile
                    )
                if hashes.numel() == 0:
                    continue

                # Steps 5-6: lazy scoring + selection.
                with _timed(profile, "predict"):
                    policy.offer(hashes, parents, gens, score_fn)

            # Empty-beam early exits (B3), per mode (see analysis: equivalent to legacy).
            sel_hashes, sel_parents, sel_gens, best_score = policy.finalize(score_fn)
            exhausted = not any_preban_candidates if mode == "simple" else sel_hashes is None or sel_hashes.numel() == 0
            if exhausted:
                if verbose >= 1:
                    print(f"Cannot find new states at step {i_step}.")
                return BeamSearchResult(False, i_step, None, debug_scores, graph.definition)

            # debug_scores per mode (legacy formulas: selected-min for
            # simple/advanced/iterated; global survivor min for iterated_batched).
            if best_score is not None:
                debug_scores[i_step] = best_score

            if verbose >= 2:
                if best_score is None:
                    if verbose_step:
                        print(f"Step {i_step}, not scored cause beam_width is big enough.")
                    else:
                        print(f"Iteration {i_step}, not scored cause beam_width is big enough.")
                elif verbose_step:
                    print(f"Step {i_step}, best score: {best_score:.2f}.")
                else:
                    print(f"Iteration {i_step}, best score {best_score}.")

            # Step 7: lazy survivor materialization (one gather of B x n).
            with _timed(profile, "moves"):
                beam_states = self._materialize(beam_states, sel_parents, sel_gens)
            beam_hashes = sel_hashes

            if return_path:
                # Order-insensitive: graph.restore_path re-sorts layers itself.
                restore_path_hashes.append(beam_hashes.to(path_device))

            if memory_cleanup:
                graph.free_memory()

            if verbose >= 10 and (i_step - 1) % 10 == 0:
                print(f"Step {i_step}, beam size: {beam_states.shape[0]}.")

            if profile is not None:
                _cuda_sync()
                print(profile.format_line(i_step, t0))

        # Path not found.
        if verbose >= 1:
            print(f"Path not found after {max_steps} steps.")
        return _finalize_not_found(max_steps, debug_scores, graph)

    # -- Mode entry points (legacy signatures; called by FastBeamSearchAlgorithm) --

    def search_simple(
        self,
        start_state: Any,
        *,
        predictor: Optional[Predictor] = None,
        beam_width: int = 1000,
        max_steps: int = 1000,
        return_path: bool = False,
        path_device: Any = "auto",
        hashed_neigbourhood: Optional[Any] = None,
        memory_cleanup: bool = False,
        verbose: int = 0,
    ) -> BeamSearchResult:
        """Fast-path replacement for legacy ``search_simple`` (destination forced to central)."""
        # Legacy contract (A1-aware): simple mode forces the central state and
        # ignores any destination_state.
        return self._run(
            "simple",
            start_state=start_state,
            destination_state=self.graph.central_state,
            predictor=predictor,
            beam_width=beam_width,
            max_steps=max_steps,
            return_path=return_path,
            path_device=path_device,
            history_depth=0,
            hashed_neigbourhood=hashed_neigbourhood,
            memory_cleanup=memory_cleanup,
            verbose=verbose,
        )

    def search_advanced(
        self,
        start_state: Any,
        destination_state: Optional[Any] = None,
        *,
        predictor: Optional[Predictor] = None,
        beam_width: int = 1000,
        max_steps: int = 1000,
        return_path: bool = False,
        path_device: Any = "auto",
        history_depth: int = 0,
        hashed_neigbourhood: Optional[Any] = None,
        memory_cleanup: bool = False,
        verbose: int = 0,
    ) -> BeamSearchResult:
        """Fast-path replacement for legacy ``search_advanced``."""
        return self._run(
            "advanced",
            start_state=start_state,
            destination_state=destination_state,
            predictor=predictor,
            beam_width=beam_width,
            max_steps=max_steps,
            return_path=return_path,
            path_device=path_device,
            history_depth=history_depth,
            hashed_neigbourhood=hashed_neigbourhood,
            memory_cleanup=memory_cleanup,
            verbose=verbose,
        )

    def search_iterated(
        self,
        start_state: Any,
        destination_state: Optional[Any] = None,
        *,
        predictor: Optional[Predictor] = None,
        beam_width: int = 1000,
        max_steps: int = 1000,
        return_path: bool = False,
        path_device: Any = "auto",
        history_depth: int = 0,
        hashed_neigbourhood: Optional[Any] = None,
        memory_cleanup: bool = False,
        verbose: int = 0,
    ) -> BeamSearchResult:
        """Fast-path replacement for legacy ``search_iterated``."""
        return self._run(
            "iterated",
            start_state=start_state,
            destination_state=destination_state,
            predictor=predictor,
            beam_width=beam_width,
            max_steps=max_steps,
            return_path=return_path,
            path_device=path_device,
            history_depth=history_depth,
            hashed_neigbourhood=hashed_neigbourhood,
            memory_cleanup=memory_cleanup,
            verbose=verbose,
        )

    def search_iterated_batched(
        self,
        start_state: Any,
        destination_state: Optional[Any] = None,
        *,
        predictor: Optional[Predictor] = None,
        beam_width: int = 1000,
        max_steps: int = 1000,
        return_path: bool = False,
        path_device: Any = "auto",
        history_depth: int = 0,
        hashed_neigbourhood: Optional[Any] = None,
        memory_cleanup: bool = False,
        verbose: int = 0,
    ) -> BeamSearchResult:
        """Fast-path replacement for legacy ``search_iterated_batched``.

        The legacy CUDA memory gate (raises ``MemoryError``) is replaced by
        chunk-bounded accounting: beyond the candidate-buffer budget the engine
        uses per-generator (iterated) selection instead of failing.
        """
        return self._run(
            "iterated_batched",
            start_state=start_state,
            destination_state=destination_state,
            predictor=predictor,
            beam_width=beam_width,
            max_steps=max_steps,
            return_path=return_path,
            path_device=path_device,
            history_depth=history_depth,
            hashed_neigbourhood=hashed_neigbourhood,
            memory_cleanup=memory_cleanup,
            verbose=verbose,
        )


# -----------------------------------------------------------------------------
# Engine factory with cached probe.
# -----------------------------------------------------------------------------

_PROBE_CACHE: dict[str, bool] = {}


def _probe_ok() -> bool:
    if "ok" not in _PROBE_CACHE:
        from ._probe import run_probe  # pylint: disable=import-outside-toplevel

        _PROBE_CACHE["ok"] = run_probe().ok
    return _PROBE_CACHE["ok"]


# Per-graph engine cache: FastBeamEngine is stateless across searches (all of
# its attributes are immutable graph-derived constants: VH matrix, permutations,
# central_by_gen; per-step state lives in locals), so one instance can serve
# every search on the same graph. Weak keys let graphs be garbage-collected.
# Values may be None (negative cache: e.g. matrix groups) — only keys are weak.
# Rationale (Kaggle cpu-bench): ms-scale rows (lrx16 etc.) were 0.74x-0.91x
# because per-call create_engine setup (VH build, argsort, central_by_gen)
# dominated searches that finish in 6-16 steps.
_ENGINE_CACHE: "weakref.WeakKeyDictionary[CayleyGraph, Optional[FastBeamEngine]]" = weakref.WeakKeyDictionary()


def create_engine(graph: "CayleyGraph", _mode: str, _predictor: Any) -> Optional[FastBeamEngine]:
    """Create (or fetch cached) engine for this graph, or ``None`` if unavailable.

    Gate (plan, section "Engine"): probe passed AND permutation group AND
    ``string_encoder is None`` AND dot-product (non-identity) hasher. The mode
    and predictor do not affect availability; they are passed per call.
    """
    if not _probe_ok():
        return None
    if graph in _ENGINE_CACHE:
        return _ENGINE_CACHE[graph]
    hv = build_permuted_hash_vectors(graph)
    engine = FastBeamEngine(graph, hv) if hv is not None else None
    _ENGINE_CACHE[graph] = engine
    return engine
