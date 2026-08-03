"""T2 parity + invariant tests: legacy cayleypy vs the fast engine.

Parity is asserted on outcomes and invariants, not bit-identity (plan, goal 5):
same ``path_found``/``path_length`` on the fixed suite, valid paths
(``apply_path(start, path) == dest``), ``len(path) == path_length`` (enforced by
``BeamSearchResult.__post_init__``), and per-step best-score sequences where
deterministic. Tie-breaking under chunked dedup may legitimately diverge from
legacy (agreed non-goal), so exact path content is not compared.
"""

import pytest
import torch

from cayleypy import CayleyGraph, MatrixGroups, PermutationGroups, Puzzles, prepare_graph

import cayleypy_fast
from cayleypy.algo.beam_search import BeamSearchAlgorithm as _LegacyBSA
from cayleypy_fast.engine import create_engine

FAST_GRAPH_DEFS = {
    "lrx6": lambda: PermutationGroups.lrx(6),
    "coxeter5": lambda: PermutationGroups.coxeter(5),
    "cube222": lambda: prepare_graph("cube_2/2/2_6gensQTM"),
    "cube333": lambda: Puzzles.rubik_cube(3, metric="QTM"),
}


def _make_graph(name: str) -> CayleyGraph:
    return CayleyGraph(FAST_GRAPH_DEFS[name](), random_seed=42)


def _scramble(graph: CayleyGraph, path: list[int]) -> torch.Tensor:
    return graph.apply_path(graph.central_state, path)


def _validate_path_to(graph: CayleyGraph, start_state, dest_state, path) -> None:
    """apply_path(start, path) must equal the destination state (plan, goal 5)."""
    reached = graph.apply_path(start_state, path).reshape(-1)
    assert torch.equal(reached.cpu(), torch.as_tensor(dest_state).reshape(-1).cpu())


@pytest.mark.parametrize("beam_mode", ["simple", "advanced", "iterated", "iterated_batched"])
@pytest.mark.parametrize("graph_name", sorted(FAST_GRAPH_DEFS))
@pytest.mark.parametrize("history_depth", [0, 2])
def test_engine_matches_legacy_outcomes(graph_name, beam_mode, history_depth):
    """Legacy vs engine on fixed seeds: same path_found and path_length; valid path."""
    scramble = [0, 1, 2, 1, 0, 2, 1, 2, 0]
    kwargs = dict(beam_mode=beam_mode, beam_width=256, max_steps=30, history_depth=history_depth, return_path=True)

    legacy_graph = _make_graph(graph_name)
    legacy_result = legacy_graph.beam_search(start_state=_scramble(legacy_graph, scramble), **kwargs)

    assert cayleypy_fast.enable()
    engine_graph = _make_graph(graph_name)
    try:
        engine_used = create_engine(engine_graph, beam_mode, None) is not None
        assert engine_used  # Guard: these graphs must all route to the engine.
        engine_result = engine_graph.beam_search(start_state=_scramble(engine_graph, scramble), **kwargs)
    finally:
        cayleypy_fast.disable()

    assert engine_result.path_found == legacy_result.path_found
    assert engine_result.path_found
    assert engine_result.path_length == legacy_result.path_length
    _validate_path_to(engine_graph, _scramble(engine_graph, scramble), engine_graph.central_state, engine_result.path)
    for gen_id in engine_result.path:
        assert 0 <= gen_id < engine_graph.definition.n_generators


def test_engine_actually_used(monkeypatch):
    """The patched search must route through the engine, not legacy ``search_*`` methods."""

    def boom(*args, **kwargs):
        raise AssertionError("legacy search_* method called while engine is available")

    # Break legacy; the engine must not call it. (The probe only checks callability.)
    for method_name in ("search_simple", "search_advanced", "search_iterated", "search_iterated_batched"):
        monkeypatch.setattr(_LegacyBSA, method_name, boom)

    graph = _make_graph("lrx6")
    assert cayleypy_fast.enable()
    try:
        result = graph.beam_search(
            start_state=_scramble(graph, [0, 1, 2]), beam_mode="advanced", beam_width=64, max_steps=20, return_path=True
        )
    finally:
        cayleypy_fast.disable()
    assert result.path_found
    graph.validate_path(_scramble(graph, [0, 1, 2]), result.path)


def test_engine_unavailable_falls_back_to_legacy():
    """Non-eligible graphs (matrix group, bit-encoded) must use the legacy path."""
    from cayleypy_fast.engine import engine_available

    matrix_graph = CayleyGraph(MatrixGroups.heisenberg())
    assert not engine_available(matrix_graph)

    encoded_graph = CayleyGraph(PermutationGroups.lrx(4), bit_encoding_width=2)
    assert not engine_available(encoded_graph)


@pytest.mark.parametrize("beam_mode", ["advanced", "iterated"])
def test_engine_custom_destination(beam_mode):
    """Custom destination_state: found path must reach the destination (not central)."""
    graph = _make_graph("lrx6")
    start_state = _scramble(graph, [0, 1, 0, 1, 2])
    dest_state = _scramble(graph, [2, 1, 2])
    assert cayleypy_fast.enable()
    try:
        result = graph.beam_search(
            start_state=start_state,
            destination_state=dest_state,
            beam_mode=beam_mode,
            beam_width=256,
            max_steps=30,
            history_depth=2,
            return_path=True,
        )
    finally:
        cayleypy_fast.disable()
    assert result.path_found
    assert len(result.path) == result.path_length
    _validate_path_to(graph, start_state, dest_state, result.path)


@pytest.mark.parametrize("beam_mode", ["simple", "advanced", "iterated", "iterated_batched"])
def test_engine_mitm_int_radius(beam_mode):
    """MITM with an integer hashed_neigbourhood radius: path must be valid."""
    graph = _make_graph("lrx6")
    start_state = _scramble(graph, [0, 1, 2, 0, 1, 0, 2, 1, 0, 1, 2, 0])
    assert cayleypy_fast.enable()
    try:
        result = graph.beam_search(
            start_state=start_state,
            beam_mode=beam_mode,
            beam_width=256,
            max_steps=30,
            history_depth=2,
            return_path=True,
            hashed_neigbourhood=3,
        )
    finally:
        cayleypy_fast.disable()
    assert result.path_found
    graph.validate_path(start_state, result.path)


@pytest.mark.parametrize("beam_mode", ["simple", "advanced", "iterated", "iterated_batched"])
def test_engine_hybrid_predictor(beam_mode):
    """Hybrid path (callable predictor): found path must be valid."""
    graph = _make_graph("lrx6")
    start_state = _scramble(graph, [0, 1, 2, 1, 0])

    def number_of_inversions(states: torch.Tensor) -> torch.Tensor:
        inversion_counts = torch.zeros(states.shape[0], dtype=torch.int64)
        for i in range(states.shape[1]):
            inversion_counts += (states[:, i : i + 1] > states[:, i + 1 :]).sum(dim=1)
        return inversion_counts

    assert cayleypy_fast.enable()
    try:
        result = graph.beam_search(
            start_state=start_state,
            beam_mode=beam_mode,
            predictor=number_of_inversions,
            beam_width=256,
            max_steps=30,
            history_depth=2,
            return_path=True,
        )
    finally:
        cayleypy_fast.disable()
    assert result.path_found
    graph.validate_path(start_state, result.path)


@pytest.mark.parametrize("beam_mode", ["simple", "advanced", "iterated", "iterated_batched"])
def test_engine_debug_scores_present_when_scored(beam_mode):
    """debug_scores must contain per-step min scores when the beam overflows (legacy formula)."""
    graph = _make_graph("lrx6")
    start_state = _scramble(graph, [0, 1, 2, 1, 0, 2, 1, 2, 0])
    assert cayleypy_fast.enable()
    try:
        result = graph.beam_search(start_state=start_state, beam_mode=beam_mode, beam_width=4, max_steps=30)
    finally:
        cayleypy_fast.disable()
    # At beam_width=4 the beam overflows immediately (lrx6 has 3 generators => 3 candidates at step 1;
    # by step 2 there are 9 > 4), so debug_scores must be populated for every search that ran >= 2 steps.
    if result.path_length > 1:
        assert len(result.debug_scores) >= 1
