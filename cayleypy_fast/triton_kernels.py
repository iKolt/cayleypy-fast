"""Optional Triton tier (T3): CUDA int64 neighbor-hash kernel for the fast engine.

Why this tier exists: on CUDA, torch has no integer GEMM (cuBLAS integer is
unsupported — cayleypy's own hasher defensively try/excepts this and falls
back to a mul+sum reduction on older GPUs, ``hasher.py``). The engine's
per-step ``B x G`` neighbor-hash "matmul" therefore needs either a number
of torch ops (matmul when supported, mul+sum otherwise) or one fused Triton
kernel. This module provides the Triton rung; the dispatch ladder lives in
``engine.py`` (``resolve_backend`` / ``_hash_int64_dispatch``).

Numerical contract (bit-equality with the reference paths):
  * int64 multiply-accumulate wraps mod 2^64 (two's complement), exactly like
    torch int64 matmul, the hasher's mul+sum fallback, and the numba tier's
    int32 accumulation — so all rungs of the backend ladder are bit-equal;
    the backend-resolution probe in ``engine.resolve_backend`` asserts this
    empirically per device before selecting this tier;
  * state values are small permutation indices (0 <= s < n), so the lossless
    int8->int64 (or int64->int64) input cast happens on load.

Kernel shape: classic tiling without ``tl.dot`` (P100 is sm_60: no tensor
cores, and ``tl.dot`` does not support int64 anyway). Each program computes a
``BLOCK_B x BLOCK_G`` output tile by looping over the state dimension in
``BLOCK_K`` chunks: ``acc += sum_k x[b, k] * vh_t[g, k]`` (explicit broadcast-
multiply + ``tl.sum`` reduction, mirroring the numba tier's inner loops).

Offsets stay within int32 by construction: per-launch ``B`` is bounded by the
engine's ``_HASH_SUBTILE_ROWS = 2^18`` subtile split, so every pointer offset
is < 2^18 * max(G, n) << 2^31 even at 2^28 beams.

Fixed config heuristic instead of ``@triton.autotune``: autotune re-benchmarks
per shape and pollutes kernel-row timings on Kaggle. Two conservative sm_60
configs sized by generator count (see ``_pick_config``).

The module is import-safe without triton/CUDA: the compiled-kernel handle
stays ``None`` and the engine's ladder never selects this tier. Kill-switch
for debugging: ``CAYLEYPY_FAST_TRITON_DISABLE=1``.
"""

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .engine import PermutedHashVectors

_ENV_DISABLE = "CAYLEYPY_FAST_TRITON_DISABLE"
_DISABLED_VALUES = {"1", "true", "yes", "on"}

try:
    import triton  # pylint: disable=import-error
    import triton.language as tl  # pylint: disable=import-error
except ImportError:  # pragma: no cover - triton is optional (CPU-only installs)
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

# Module-level handle assigned conditionally below (see if-block).
# pylint: disable=invalid-name
_TRITON_KERNEL_INT64 = None

if triton is not None and tl is not None:

    @triton.jit
    def _hash_neighbors_int64_jit(
        states_ptr,
        vh_t_ptr,
        out_ptr,
        n_rows,
        n_gens,
        state_size,
        block_b: tl.constexpr,
        block_g: tl.constexpr,
        block_k: tl.constexpr,
    ):
        """out[i, g] = sum_k int64(states[i, k]) * vh_t[g, k] with mod-2^64 wraparound.

        All tensors are contiguous row-major: states (n_rows, state_size) int8
        or int64, vh_t (n_gens, state_size) int64, out (n_rows, n_gens) int64.
        """
        pid_b = tl.program_id(axis=0)
        pid_g = tl.program_id(axis=1)
        offs_b = pid_b * block_b + tl.arange(0, block_b)
        offs_g = pid_g * block_g + tl.arange(0, block_g)
        b_mask = offs_b < n_rows
        g_mask = offs_g < n_gens
        acc = tl.zeros((block_b, block_g), dtype=tl.int64)
        for k0 in range(0, state_size, block_k):
            offs_k = k0 + tl.arange(0, block_k)
            k_mask = offs_k < state_size
            x = tl.load(
                states_ptr + offs_b[:, None] * state_size + offs_k[None, :],
                mask=b_mask[:, None] & k_mask[None, :],
                other=0,
            )
            v = tl.load(
                vh_t_ptr + offs_g[:, None] * state_size + offs_k[None, :],
                mask=g_mask[:, None] & k_mask[None, :],
                other=0,
            )
            # int64 multiply-add wraps mod 2^64 (two's complement), matching the
            # torch/numba reference paths.
            acc += tl.sum(x.to(tl.int64)[:, None, :] * v[None, :, :], axis=2)
        tl.store(
            out_ptr + offs_b[:, None] * n_gens + offs_g[None, :],
            acc,
            mask=b_mask[:, None] & g_mask[None, :],
        )

    _TRITON_KERNEL_INT64 = _hash_neighbors_int64_jit


def triton_available() -> bool:
    """Whether the triton tier can be used (importable, CUDA present, not env-disabled)."""
    if _TRITON_KERNEL_INT64 is None or not torch.cuda.is_available():
        return False
    return os.environ.get(_ENV_DISABLE, "").strip().lower() not in _DISABLED_VALUES


def _pick_config(n_rows: int, n_generators: int) -> tuple[int, int, int, int]:
    """Fixed (BLOCK_B, BLOCK_G, BLOCK_K, num_warps) by shape bucket (no autotune).

    Conservative for sm_60 (Tesla P100): the 3D broadcast-product temp is
    BLOCK_B * BLOCK_G * BLOCK_K int64 lanes; these sizes keep register pressure
    within sm_60 limits. Two buckets suffice: lrx graphs have 3 generators
    (wide row blocks amortize the tiny G loop), cubes have 12-48.
    """
    if n_generators <= 8:
        return (128 if n_rows >= 2**14 else 64), 4, 32, 4
    return 64, 4, 32, 4


def _triton_vh_t(hv: "PermutedHashVectors") -> torch.Tensor:
    """Transposed contiguous ``(G, n)`` int64 copy of ``hv.vh`` (cached on the instance)."""
    cached = hv._triton_vh_t  # pylint: disable=protected-access
    if cached is not None:
        return cached
    vh_t = hv.vh.t().contiguous()
    hv._triton_vh_t = vh_t  # pylint: disable=protected-access
    return vh_t


def hash_neighbors_int64_triton(hv: "PermutedHashVectors", states: torch.Tensor) -> torch.Tensor:
    """Triton int64 hash of all ``B x G`` neighbors; bit-equal to the reference paths.

    :param hv: int64 permuted hash vectors (``(n, G)`` int64, CUDA).
    :param states: ``(B, n)`` CUDA states (int8/int64 small integer values; the
        kernel casts to int64 on load, so no cast temp is materialized).
    :return: ``(B, G)`` int64 hashes.
    """
    assert not hv.dual_int32, "triton kernel handles the int64 path only"
    assert states.is_cuda, "triton tier requires CUDA states"
    n_rows, state_size = states.shape
    vh_t = _triton_vh_t(hv)
    n_generators = vh_t.shape[0]
    out = torch.empty((n_rows, n_generators), dtype=torch.int64, device=states.device)
    if n_rows == 0:
        return out
    states_c = states.contiguous()
    block_b, block_g, block_k, num_warps = _pick_config(n_rows, n_generators)
    grid = (triton.cdiv(n_rows, block_b), triton.cdiv(n_generators, block_g))
    assert _TRITON_KERNEL_INT64 is not None
    _TRITON_KERNEL_INT64[grid](  # pylint: disable=not-callable
        states_c,
        vh_t,
        out,
        n_rows,
        n_generators,
        state_size,
        block_b=block_b,
        block_g=block_g,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out
