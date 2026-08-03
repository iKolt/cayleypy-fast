"""T1 property tests: engine matmul hashes must bit-match legacy hashing.

For permutation groups with the dot-product hasher, hashing a neighbor is
linear: ``hash(g(s)) = s . (v o p_g^{-1})``. These tests verify that the
engine's permuted hash vectors (derived from the LIVE ``graph.hasher``) produce
hashes bit-identical to ``hasher.make_hashes`` run on the materialized neighbor
states — the foundation of never materializing the ``B x G`` neighbor states
(plan, "Key technical insight").
"""

import pytest
import torch

from cayleypy import CayleyGraph, CayleyGraphDef, MatrixGroups, PermutationGroups, Puzzles, prepare_graph

from cayleypy_fast.engine import (
    PermutedHashVectors,
    _permute_vector,
    build_permuted_hash_vectors,
    engine_available,
    hash_neighbors_tiled,
)

TILE = 512

# Sampled permutation graphs (plan, T1): lrx 4/8/16/32, cube222/333/444/555,
# plus a few more PermutationGroups entries.
_GRAPH_FACTORIES = {
    "lrx4": lambda: PermutationGroups.lrx(4),
    "lrx8": lambda: PermutationGroups.lrx(8),
    "lrx16": lambda: PermutationGroups.lrx(16),
    "lrx32": lambda: PermutationGroups.lrx(32),
    "cube222": lambda: prepare_graph("cube_2/2/2_6gensQTM"),
    "cube333": lambda: Puzzles.rubik_cube(3, metric="QTM"),
    "cube444": lambda: Puzzles.rubik_cube(4, metric="QTM"),
    "cube555": lambda: Puzzles.rubik_cube(5, metric="QTM"),
    "pancake6": lambda: PermutationGroups.pancake(6),
    "top_spin6": lambda: PermutationGroups.top_spin(6),
    "coxeter5": lambda: PermutationGroups.coxeter(5),
    "full_reversals6": lambda: PermutationGroups.full_reversals(6),
}


def _valid_states(graph: CayleyGraph, width: int, length: int = 8) -> torch.Tensor:
    """Generate ``width`` valid (reachable, dtype-safe) states via random walks.

    ``random_walks`` returns a step-major 2D tensor of shape (length * width, n);
    the final states of all walks are the last ``width`` rows.
    """
    states, _ = graph.random_walks(width=width, length=length)
    return states[(length - 1) * width :, :]


@pytest.mark.parametrize("graph_name", sorted(_GRAPH_FACTORIES))
@pytest.mark.parametrize("dtype", [torch.int8, torch.int64])
def test_engine_hashes_bitmatch_make_hashes(graph_name, dtype):
    """Engine matmul hashes == make_hashes(materialized neighbors), on the live hasher.

    On CPU this exercises the dual-int32 hasher path; on CUDA the int64 path is
    picked up automatically (the hasher then has ``vec_hasher``).
    """
    graph = CayleyGraph(_GRAPH_FACTORIES[graph_name](), dtype=dtype, random_seed=42)
    hv = build_permuted_hash_vectors(graph)
    assert hv is not None
    n_generators = graph.definition.n_generators

    states = graph.encode_states(_valid_states(graph, TILE))

    # Legacy ground truth: materialize neighbors then hash (gen-major layout).
    neighbors = graph.get_neighbors(states)
    legacy_hashes = graph.hasher.make_hashes(neighbors).reshape(n_generators, TILE)

    engine_hashes = hash_neighbors_tiled(hv, states)
    assert engine_hashes.dtype == torch.int64
    assert engine_hashes.shape == (TILE, n_generators)
    # Transpose to gen-major to compare against get_neighbors' (G*B,) layout.
    assert torch.equal(engine_hashes.t(), legacy_hashes)


def test_vh_int64_matches_explicit_materialized_hashing():
    """int64 VH construction math (GPU path) verified on CPU against mul+sum.

    Integer wrap-around arithmetic is associativity/commutativity exact mod 2^64,
    so ``sum(states * v)`` (the "older GPU" reference) and the matmul are
    bit-identical. This validates the VH construction itself, path-independent.
    """
    graph = CayleyGraph(PermutationGroups.lrx(8), random_seed=7)
    n = graph.definition.state_size
    inv_perms = torch.argsort(graph.permutations_torch, dim=1)
    torch.manual_seed(7)
    v64 = torch.randint(-(2**62), 2**62, (n,), dtype=torch.int64)

    vh = _permute_vector(v64, inv_perms, dual_int32=False)
    assert vh.shape == (n, graph.definition.n_generators)

    states = graph.encode_states(_valid_states(graph, width=256, length=6))
    neighbors = graph.get_neighbors(states)
    reference = torch.sum(neighbors.to(torch.int64) * v64, dim=1).reshape(graph.definition.n_generators, 256)

    engine_hashes = hash_neighbors_tiled(PermutedHashVectors(vh=vh, dual_int32=False), states)
    assert torch.equal(engine_hashes.t(), reference)


def test_build_returns_none_for_matrix_groups():
    graph = CayleyGraph(MatrixGroups.heisenberg())
    assert build_permuted_hash_vectors(graph) is None
    assert not engine_available(graph)


def test_build_returns_none_for_bit_encoded_states():
    graph = CayleyGraph(PermutationGroups.lrx(4), bit_encoding_width=2)
    assert graph.string_encoder is not None
    assert build_permuted_hash_vectors(graph) is None
    assert not engine_available(graph)


def test_build_returns_none_for_identity_hasher():
    definition = CayleyGraphDef.create(generators=[[0]], generator_names=["e"], central_state=[0], name="trivial")
    graph = CayleyGraph(definition)
    assert graph.hasher.is_identity
    assert build_permuted_hash_vectors(graph) is None
    assert not engine_available(graph)
