"""T3 triton tier tests: backend ladder, bit-equality, kill-switch, demotion.

CUDA tests are collected-but-skipped on CPU-only machines and run inside the
Kaggle perf kernel's stage-0 (``pytest tests/test_triton_tier.py`` in a clone
of the repo on the GPU runner). CPU-safe sections validate the mul+sum floor
rung and the demotion machinery without a GPU.

Ladder recap (see ``engine.resolve_backend``): triton -> matmul -> mulsum; all
int64 rungs are bit-equal (int64 mul/add wrap mod 2^64), so mid-search demotion
must never change search semantics — only emit a warning and recompute.
"""

import warnings

import pytest
import torch

import cayleypy_fast
from cayleypy import CayleyGraph, PermutationGroups, Puzzles, prepare_graph
from cayleypy_fast import engine as engine_module
from cayleypy_fast.engine import (
    PermutedHashVectors,
    build_permuted_hash_vectors,
    create_engine,
    hash_neighbors_mulsum,
    hash_neighbors_tiled,
    resolve_backend,
)

MINIGRAPH_FACTORIES = {
    "lrx8": lambda: PermutationGroups.lrx(8),
    "lrx16": lambda: PermutationGroups.lrx(16),
    "cube222": lambda: prepare_graph("cube_2/2/2_6gensQTM"),
}


def _int64_hv(graph: CayleyGraph) -> PermutedHashVectors:
    """An int64 (GPU-style) PermutedHashVectors instance derived from a live graph."""
    inv_perms = torch.argsort(graph.permutations_torch, dim=1)
    n = inv_perms.shape[1]
    torch.manual_seed(7)
    v64 = torch.randint(-(2**62), 2**62, (n,), dtype=torch.int64, device=graph.device)
    from cayleypy_fast.engine import _permute_vector  # pylint: disable=import-outside-toplevel

    return PermutedHashVectors(_permute_vector(v64, inv_perms, dual_int32=False), dual_int32=False)


def _cpu_graph(name: str) -> CayleyGraph:
    """CPU-forced graph: the *_cpu tests must stay hermetic even on GPU runners
    (device defaults to 'auto')."""
    return CayleyGraph(MINIGRAPH_FACTORIES[name](), random_seed=42, device="cpu")


# -----------------------------------------------------------------------------
# CPU-safe ladder tests (run locally).
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("graph_name", sorted(MINIGRAPH_FACTORIES))
@pytest.mark.parametrize("dtype", [torch.int8, torch.int64])
def test_mulsum_bitmatch_matmul_cpu(graph_name, dtype):
    """The mul+sum floor rung is bit-equal to torch int64 matmul (graph-realistic shapes)."""
    graph = _cpu_graph(graph_name)
    hv = _int64_hv(graph)
    assert not hv.dual_int32
    states, _ = graph.random_walks(width=256, length=8)
    states = states.reshape(-1, graph.encoded_state_size).to(dtype)

    reference = states.to(torch.int64) @ hv.vh  # (B, G) int64 torch matmul.
    assert torch.equal(hash_neighbors_mulsum(hv, states), reference)
    assert torch.equal(hash_neighbors_tiled(hv, states), reference)  # backend None -> matmul rung.


def test_mulsum_bitmatch_matmul_wraparound_cpu():
    """Bit-equality under heavy int64 wraparound: huge state values x huge vectors."""
    torch.manual_seed(7)
    n, g_count, rows = 37, 5, 513
    vh = torch.randint(-(2**62), 2**62, (n, g_count), dtype=torch.int64)
    hv = PermutedHashVectors(vh, dual_int32=False)
    states = torch.randint(0, 2**40, (rows, n), dtype=torch.int64)
    assert torch.equal(hash_neighbors_mulsum(hv, states), states.to(torch.int64) @ vh)


def test_mulsum_zero_rows_cpu():
    hv = PermutedHashVectors(torch.randint(-(2**62), 2**62, (4, 3), dtype=torch.int64), dual_int32=False)
    out = hash_neighbors_mulsum(hv, torch.empty((0, 4), dtype=torch.int8))
    assert out.shape == (0, 3) and out.dtype == torch.int64


def test_resolve_backend_noop_on_cpu():
    """CPU instances keep backend=None (dual-int32 numba/torch dispatch untouched)."""
    graph = _cpu_graph("lrx8")
    hv = build_permuted_hash_vectors(graph)
    assert hv is not None and hv.dual_int32
    resolve_backend(hv)
    assert hv.backend is None

    hv64 = _int64_hv(graph)
    resolve_backend(hv64)
    assert hv64.backend is None  # CPU int64: historical torch matmul path.


def test_matmul_failure_demotes_to_mulsum_cpu(monkeypatch):
    """A matmul-rung runtime failure warns once, demotes to mulsum, and recomputes."""
    graph = _cpu_graph("lrx8")
    hv = _int64_hv(graph)
    states = torch.randint(0, 8, (17, 8), dtype=torch.int8)
    reference = hash_neighbors_mulsum(hv, states)

    hv.backend = "matmul"
    monkeypatch.setattr(
        engine_module, "_hash_via_matmul", lambda hv_, states_: (_ for _ in ()).throw(RuntimeError("no integer GEMM"))
    )
    with pytest.warns(RuntimeWarning, match="demoting to 'mulsum'"):
        out = hash_neighbors_tiled(hv, states)
    assert hv.backend == "mulsum"
    assert torch.equal(out, reference)

    # Second call uses the demoted backend directly: no more warnings, same result.
    monkeypatch.undo()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out2 = hash_neighbors_tiled(hv, states)
    assert not [w for w in record if "demoting" in str(w.message)]
    assert torch.equal(out2, reference)


def test_triton_failure_demotes_then_bitmatch_cpu(monkeypatch):
    """Simulated triton-rung failure (no GPU needed): demote to matmul, recompute, bit-equal."""
    graph = _cpu_graph("lrx8")
    hv = _int64_hv(graph)
    states = torch.randint(0, 8, (17, 8), dtype=torch.int8)
    reference = states.to(torch.int64) @ hv.vh

    hv.backend = "triton"
    monkeypatch.setattr(
        engine_module,
        "_hash_via_triton",
        lambda hv_, states_: (_ for _ in ()).throw(RuntimeError("kernel launch boom")),
    )
    with pytest.warns(RuntimeWarning, match="demoting to 'matmul'"):
        out = hash_neighbors_tiled(hv, states)
    assert hv.backend == "matmul"
    assert torch.equal(out, reference)


# -----------------------------------------------------------------------------
# CUDA tests (skipped locally; executed on the Kaggle perf kernel stage 0).
# -----------------------------------------------------------------------------

_CUDA = torch.cuda.is_available()
_CUDA_GRAPHS = {
    "lrx8": lambda: PermutationGroups.lrx(8),
    "lrx16": lambda: PermutationGroups.lrx(16),
    "lrx32": lambda: PermutationGroups.lrx(32),
    "cube222": lambda: prepare_graph("cube_2/2/2_6gensQTM"),
    "cube333": lambda: Puzzles.rubik_cube(3, metric="QTM"),
}


def _triton_available_now() -> bool:
    from cayleypy_fast.triton_kernels import triton_available  # pylint: disable=import-outside-toplevel

    return triton_available()


@pytest.mark.skipif(not _CUDA, reason="CUDA-only tier test")
def test_triton_backend_selected_on_cuda():
    """On a CUDA box with triton installed, the ladder must resolve to the triton rung."""
    if not _triton_available_now():
        pytest.skip("triton unavailable on this CUDA box")
    graph = CayleyGraph(PermutationGroups.lrx(8), random_seed=42, device="cuda")
    engine = create_engine(graph, "simple", None)
    assert engine is not None
    assert engine.hv.backend == "triton"


@pytest.mark.skipif(not _CUDA, reason="CUDA-only tier test")
@pytest.mark.parametrize("graph_name", sorted(_CUDA_GRAPHS))
@pytest.mark.parametrize("dtype", [torch.int8, torch.int64])
def test_engine_hashes_bitmatch_make_hashes_cuda(graph_name, dtype):
    """Engine (backend-ladder) hashes bit-match hasher.make_hashes on materialized neighbors."""
    graph = CayleyGraph(_CUDA_GRAPHS[graph_name](), dtype=dtype, random_seed=42, device="cuda")
    engine = create_engine(graph, "simple", None)
    assert engine is not None
    n_generators = graph.definition.n_generators

    width = 512
    states, _ = graph.random_walks(width=width, length=8)
    states = states.reshape(-1, graph.encoded_state_size)[-width:]
    states_enc = graph.encode_states(states)

    neighbors = graph.get_neighbors(states_enc)
    legacy_hashes = graph.hasher.make_hashes(neighbors).reshape(n_generators, width)
    engine_hashes = hash_neighbors_tiled(engine.hv, states_enc)
    assert engine_hashes.dtype == torch.int64
    assert torch.equal(engine_hashes.t(), legacy_hashes)


@pytest.mark.skipif(not _CUDA, reason="CUDA-only tier test")
def test_triton_kill_switch_forces_fallback_cuda(monkeypatch):
    """CAYLEYPY_FAST_TRITON_DISABLE=1 keeps the engine correct on a lower rung."""
    monkeypatch.setenv("CAYLEYPY_FAST_TRITON_DISABLE", "1")
    assert not _triton_available_now()
    graph = CayleyGraph(PermutationGroups.lrx(8), random_seed=42, device="cuda")
    engine = create_engine(graph, "simple", None)
    assert engine is not None
    assert engine.hv.backend in ("matmul", "mulsum")

    width = 256
    states, _ = graph.random_walks(width=width, length=8)
    states_enc = graph.encode_states(states.reshape(-1, graph.encoded_state_size)[-width:])
    neighbors = graph.get_neighbors(states_enc)
    legacy_hashes = graph.hasher.make_hashes(neighbors).reshape(graph.definition.n_generators, width)
    assert torch.equal(hash_neighbors_tiled(engine.hv, states_enc).t(), legacy_hashes)


@pytest.mark.skipif(not _CUDA, reason="CUDA-only tier test")
def test_mid_search_triton_demotion_keeps_parity(monkeypatch):
    """A triton kernel failing on hash call #2 mid-search: warn, demote, IDENTICAL outcome.

    All rungs are bit-equal, so the demoted engine's trajectory must match the
    legacy run exactly (outcome parity is the legal assert; here we additionally
    expect path_length equality since the divergence source is absent).
    """
    if not _triton_available_now():
        pytest.skip("triton unavailable on this CUDA box")
    graph = CayleyGraph(PermutationGroups.lrx(8), random_seed=42, device="cuda")
    start_state = graph.apply_path(graph.central_state, [0, 1, 2, 1, 0, 2, 1, 2, 0])
    kwargs = dict(beam_mode="advanced", beam_width=512, max_steps=30, history_depth=2, return_path=True)

    legacy_result = graph.beam_search(start_state=start_state, **kwargs)

    assert cayleypy_fast.enable()
    try:
        real_triton = engine_module._hash_via_triton
        state = {"calls": 0}

        def failing_on_second(hv, states):
            state["calls"] += 1
            if state["calls"] == 2:
                raise RuntimeError("simulated triton kernel failure")
            return real_triton(hv, states)

        with monkeypatch.context() as m:
            # Engine resolution must succeed before the monkeypatch (probes use the same fn).
            engine = create_engine(graph, "advanced", None)
            assert engine is not None and engine.hv.backend == "triton"
            m.setattr(engine_module, "_hash_via_triton", failing_on_second)
            with pytest.warns(RuntimeWarning, match="demoting"):
                engine_result = graph.beam_search(start_state=start_state, **kwargs)
        assert state["calls"] == 2  # One success, one failure (demotion is sticky).
        assert engine.hv.backend != "triton"
    finally:
        cayleypy_fast.disable()

    assert legacy_result.path_found
    assert engine_result.path_found == legacy_result.path_found
    assert engine_result.path_length == legacy_result.path_length
    graph.validate_path(start_state, engine_result.path)
