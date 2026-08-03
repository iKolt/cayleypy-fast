"""T4 numba tier tests: bit-equality, dispatch, and speed guard.

The numba kernel must produce hashes bit-identical to the torch reference path
(which is itself pinned to ``hasher.make_hashes`` by test_hash_parity.py), on
every supported permutation graph and dtype. Additional tests cover the
kill-switch env var, the no-numba fallback, and a coarse speed guard so the
tier is never silently slower than torch on a benchmark-scale tile.
"""

import time

import pytest
import torch

from cayleypy import CayleyGraph, PermutationGroups, Puzzles, prepare_graph

from cayleypy_fast import numba_kernels
from cayleypy_fast.engine import build_permuted_hash_vectors, hash_neighbors_tiled
from cayleypy_fast.numba_kernels import hash_neighbors_dual_int32_numba, numba_available

# Self-contained graph sample (same coverage spirit as test_hash_parity.py):
# small + benchmark-scale permutation graphs.
_GRAPH_FACTORIES = {
    "lrx8": lambda: PermutationGroups.lrx(8),
    "lrx16": lambda: PermutationGroups.lrx(16),
    "lrx32": lambda: PermutationGroups.lrx(32),
    "cube222": lambda: prepare_graph("cube_2/2/2_6gensQTM"),
    "cube333": lambda: Puzzles.rubik_cube(3, metric="QTM"),
    "cube555": lambda: Puzzles.rubik_cube(5, metric="QTM"),
}


def _valid_states(graph: CayleyGraph, width: int, length: int = 8, dtype=None) -> torch.Tensor:
    states, _ = graph.random_walks(width=width, length=length)
    out = states[(length - 1) * width :, :]
    return out if dtype is None else out.to(dtype)


pytestmark = pytest.mark.skipif(not numba_available(), reason="numba tier unavailable (not installed or env-disabled)")


def _torch_reference(hv, states):
    """The pure-torch dual-int32 reference path (pre-T4 body of hash_neighbors_tiled)."""
    h32 = states.to(torch.int32) @ hv.vh
    n_gen = hv.vh.shape[1] // 2
    return h32.view(-1, n_gen, 2).view(torch.int64).reshape(-1, n_gen)


@pytest.mark.parametrize("graph_name", sorted(_GRAPH_FACTORIES))
@pytest.mark.parametrize("dtype", [torch.int8, torch.int64])
def test_numba_hashes_bitmatch_torch_reference(graph_name, dtype):
    graph = CayleyGraph(_GRAPH_FACTORIES[graph_name](), random_seed=42)
    hv = build_permuted_hash_vectors(graph)
    if hv is None or not hv.dual_int32:  # pragma: no cover - all CPU perm graphs are dual_int32
        pytest.skip("dual-int32 hasher path unavailable for this graph/device")
    states = _valid_states(graph, width=64, dtype=dtype)
    assert torch.equal(hash_neighbors_dual_int32_numba(hv, states), _torch_reference(hv, states))


def test_dispatch_uses_numba_by_default():
    graph = CayleyGraph(_GRAPH_FACTORIES["lrx16"](), random_seed=42)
    hv = build_permuted_hash_vectors(graph)
    assert hv is not None and hv.dual_int32
    states = _valid_states(graph, width=64)
    assert torch.equal(hash_neighbors_tiled(hv, states), hash_neighbors_dual_int32_numba(hv, states))


def test_numba_disable_env_var(monkeypatch):
    monkeypatch.setenv("CAYLEYPY_FAST_NUMBA_DISABLE", "1")
    assert not numba_available()
    monkeypatch.delenv("CAYLEYPY_FAST_NUMBA_DISABLE")
    assert numba_available()


def test_numba_fallback_killswitch(monkeypatch):
    """hash_neighbors_tiled must fall back to the torch path when env-disabled."""
    graph = CayleyGraph(_GRAPH_FACTORIES["lrx16"](), random_seed=42)
    hv = build_permuted_hash_vectors(graph)
    assert hv is not None and hv.dual_int32
    states = _valid_states(graph, width=64)

    monkeypatch.setenv("CAYLEYPY_FAST_NUMBA_DISABLE", "1")
    torch_out = hash_neighbors_tiled(hv, states)
    assert torch.equal(torch_out, _torch_reference(hv, states))
    monkeypatch.delenv("CAYLEYPY_FAST_NUMBA_DISABLE")
    assert torch.equal(hash_neighbors_tiled(hv, states), torch_out)


def test_numba_vh_t_caching():
    """The transposed int32 matrix is cached on the PermutedHashVectors instance."""
    graph = CayleyGraph(_GRAPH_FACTORIES["lrx16"](), random_seed=42)
    hv = build_permuted_hash_vectors(graph)
    assert hv is not None and hv.dual_int32
    assert hv._numba_vh_t is None
    states = _valid_states(graph, width=8)
    hash_neighbors_dual_int32_numba(hv, states)
    cached = hv._numba_vh_t
    assert cached is not None and cached.dtype.name == "int32"
    hash_neighbors_dual_int32_numba(hv, states)
    assert hv._numba_vh_t is cached  # same object, not rebuilt


def test_numba_speed_guard_benchmark_scale():
    """Coarse guard: numba must beat torch matmul on a benchmark-scale tile.

    Measured on laptop (Windows i5, torch 2.13 CPU): 19-29x on cube555-shape
    tiles. Assert >2x so it's robust across CI CPUs but still catches a silent
    fallback or vectorization failure.
    """
    graph = CayleyGraph(_GRAPH_FACTORIES["cube555"](), random_seed=42)
    hv = build_permuted_hash_vectors(graph)
    assert hv is not None and hv.dual_int32
    states = _valid_states(graph, width=2**14, length=4)

    # Warm-up both paths (JIT compile + threading pools).
    for _ in range(2):
        hash_neighbors_dual_int32_numba(hv, states)
        _torch_reference(hv, states)

    t0 = time.perf_counter()
    out_numba = hash_neighbors_dual_int32_numba(hv, states)
    t_numba = time.perf_counter() - t0
    t0 = time.perf_counter()
    out_torch = _torch_reference(hv, states)
    t_torch = time.perf_counter() - t0

    assert torch.equal(out_numba, out_torch)
    assert t_numba < t_torch / 2, f"numba {t_numba*1e3:.1f}ms vs torch {t_torch*1e3:.1f}ms"
