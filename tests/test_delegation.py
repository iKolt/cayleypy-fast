"""T0 characterization: the patched beam search must delegate to legacy cayleypy.

With no engine implemented yet (T0), ``FastBeamSearchAlgorithm`` is a pure
pass-through, so patched and unpatched runs must produce identical results.
This proves the patch plumbing end-to-end (plan, T0).
"""

import pytest

from cayleypy import CayleyGraph, PermutationGroups

import cayleypy_fast

SCRAMBLE_PATH = [0, 1, 2, 1, 0, 2, 1, 2, 0]
BEAM_MODES = ["simple", "advanced", "iterated", "iterated_batched"]


def _make_graph() -> CayleyGraph:
    return CayleyGraph(PermutationGroups.lrx(6), random_seed=42)


def _run(graph: CayleyGraph, beam_mode: str):
    start_state = graph.apply_path(graph.central_state, SCRAMBLE_PATH)
    return start_state, graph.beam_search(
        start_state=start_state,
        beam_mode=beam_mode,
        beam_width=16,
        max_steps=50,
        history_depth=2,
        return_path=True,
    )


@pytest.mark.parametrize("beam_mode", BEAM_MODES)
def test_patched_matches_legacy(beam_mode):
    graph = _make_graph()
    start_state, legacy_result = _run(graph, beam_mode)
    assert legacy_result.path_found
    graph.validate_path(start_state, legacy_result.path)

    assert cayleypy_fast.enable()
    patched_graph = _make_graph()
    _, patched_result = _run(patched_graph, beam_mode)

    assert patched_result.path_found == legacy_result.path_found
    assert patched_result.path_length == legacy_result.path_length
    assert patched_result.path == legacy_result.path
    assert patched_result.debug_scores == legacy_result.debug_scores


def test_wrap_routes_through_fast_class_without_patching():
    graph = _make_graph()
    start_state = graph.apply_path(graph.central_state, SCRAMBLE_PATH)
    assert not cayleypy_fast.is_enabled()
    wrapped = cayleypy_fast.wrap(graph)
    result = wrapped.beam_search(start_state=start_state, beam_width=16, max_steps=50, return_path=True)
    assert result.path_found
    graph.validate_path(start_state, result.path)
    assert wrapped.wrapped_graph is graph
    # Attribute passthrough.
    assert wrapped.definition is graph.definition
