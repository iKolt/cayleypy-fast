"""Tests for the enable()/disable() in-place patching machinery."""

import pytest

import cayleypy.algo
import cayleypy.algo.beam_search as beam_search_module
import cayleypy.cayley_graph as cayley_graph_module
import cayleypy_fast

# Bound at import time, before any test can enable the patch.
LEGACY_CLASS = beam_search_module.BeamSearchAlgorithm

PATCHED_MODULES = [cayley_graph_module, cayleypy.algo, beam_search_module]


@pytest.mark.parametrize("module", PATCHED_MODULES)
def test_enable_patches_symbol(module):
    assert cayleypy_fast.enable()
    patched = module.BeamSearchAlgorithm
    assert patched is not LEGACY_CLASS
    assert issubclass(patched, LEGACY_CLASS)
    assert patched.__module__ == "cayleypy_fast._patch"


def test_enable_is_idempotent():
    assert cayleypy_fast.enable()
    first = beam_search_module.BeamSearchAlgorithm
    assert cayleypy_fast.enable()
    assert beam_search_module.BeamSearchAlgorithm is first
    assert cayleypy_fast.is_enabled()


def test_disable_restores_symbols():
    assert cayleypy_fast.enable()
    assert cayleypy_fast.disable()
    for module in PATCHED_MODULES:
        assert module.BeamSearchAlgorithm is LEGACY_CLASS
    assert not cayleypy_fast.is_enabled()
    # A second disable is a no-op.
    assert not cayleypy_fast.disable()


def test_env_var_forces_noop(monkeypatch):
    monkeypatch.setenv(cayleypy_fast.ENV_DISABLE, "1")
    assert not cayleypy_fast.enable()
    assert not cayleypy_fast.is_enabled()
    for module in PATCHED_MODULES:
        assert module.BeamSearchAlgorithm is LEGACY_CLASS


def test_enable_noop_with_warning_when_probe_fails(monkeypatch):
    monkeypatch.delattr(beam_search_module, "_restore_path")
    with pytest.warns(RuntimeWarning, match="incompatible"):
        assert not cayleypy_fast.enable()
    assert not cayleypy_fast.is_enabled()
    for module in PATCHED_MODULES:
        assert module.BeamSearchAlgorithm is LEGACY_CLASS


# --- Size gate tests -----------------------------------------------------------
# Default gate: engine only when beam_width * n_generators >= 2**16 OR
# beam_width >= 512. Small problems must route to legacy.


def _boom_create_engine(graph, mode, predictor):  # noqa: ARG001
    raise AssertionError("engine.create_engine must not be called (size gate off)")


def _lrx_graph():
    from cayleypy import CayleyGraph, PermutationGroups

    return CayleyGraph(PermutationGroups.lrx(6), random_seed=42)


def test_size_gate_routes_small_problems_to_legacy(monkeypatch):
    """beam_width=4 on lrx6 (3 gens): 12 << 2^16 and 4 < 512 -> legacy path."""
    monkeypatch.setenv("CAYLEYPY_FAST_MIN_BG", str(2**16))
    monkeypatch.setenv("CAYLEYPY_FAST_MIN_BEAM", "512")
    monkeypatch.setattr("cayleypy_fast.engine.create_engine", _boom_create_engine)
    graph = _lrx_graph()
    assert cayleypy_fast.enable()
    try:
        result = graph.beam_search(start_state=graph.central_state, beam_width=4, max_steps=5)
    finally:
        cayleypy_fast.disable()
    assert result.path_found


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beam_width": 512},  # meets min_beam
        {"beam_width": 1000},  # meets min_beam; also no-beam_width default below
        {},  # default beam_width=1000
    ],
)
def test_size_gate_engages_engine_when_large(monkeypatch, kwargs):
    """Large (or default) beams must engage the engine; creation booms."""
    monkeypatch.delenv("CAYLEYPY_FAST_MIN_BG", raising=False)
    monkeypatch.delenv("CAYLEYPY_FAST_MIN_BEAM", raising=False)
    monkeypatch.setattr("cayleypy_fast.engine.create_engine", _boom_create_engine)
    graph = _lrx_graph()
    assert cayleypy_fast.enable()
    try:
        with pytest.raises(AssertionError, match="size gate off"):
            graph.beam_search(start_state=graph.central_state, beam_mode="advanced", max_steps=5, **kwargs)
    finally:
        cayleypy_fast.disable()


def test_size_gate_env_override_forces_engine_on(monkeypatch):
    """Setting both thresholds to 0 makes the engine engage even for tiny beams."""
    monkeypatch.setenv("CAYLEYPY_FAST_MIN_BG", "0")
    monkeypatch.setenv("CAYLEYPY_FAST_MIN_BEAM", "0")
    monkeypatch.setattr("cayleypy_fast.engine.create_engine", _boom_create_engine)
    graph = _lrx_graph()
    assert cayleypy_fast.enable()
    try:
        with pytest.raises(AssertionError, match="size gate off"):
            graph.beam_search(start_state=graph.central_state, beam_width=4, max_steps=5)
    finally:
        cayleypy_fast.disable()


def test_size_gate_invalid_env_falls_back_to_defaults(monkeypatch):
    """Non-numeric env values must not crash; defaults apply."""
    monkeypatch.setenv("CAYLEYPY_FAST_MIN_BG", "banana")
    monkeypatch.setenv("CAYLEYPY_FAST_MIN_BEAM", "n/a")
    monkeypatch.setattr("cayleypy_fast.engine.create_engine", _boom_create_engine)
    graph = _lrx_graph()
    assert cayleypy_fast.enable()
    try:
        # Small beam -> defaults gate engine off -> legacy works.
        result = graph.beam_search(start_state=graph.central_state, beam_width=4, max_steps=5)
    finally:
        cayleypy_fast.disable()
    assert result.path_found


# --- Per-graph engine cache ---------------------------------------------------


def test_engine_cached_per_graph_instances():
    """create_engine must return the SAME engine object for repeated calls on one graph.

    Kaggle cpu-bench showed 0.74x-0.91x on ms-scale rows because per-call engine
    setup (VH build, argsort, central_by_gen) dominated 6-16-step searches.
    """
    from cayleypy import CayleyGraph, MatrixGroups  # pylint: disable=import-outside-toplevel

    from cayleypy_fast import engine as engine_module  # pylint: disable=import-outside-toplevel

    graph = _lrx_graph()
    e1 = engine_module.create_engine(graph, "simple", None)
    e2 = engine_module.create_engine(graph, "advanced", None)
    assert e1 is not None
    assert e1 is e2

    # A different graph instance gets its own engine.
    other = _lrx_graph()
    e3 = engine_module.create_engine(other, "simple", None)
    assert e3 is not None
    assert e3 is not e1

    # Negative cache: matrix groups consistently yield None.
    matrix_graph = CayleyGraph(MatrixGroups.heisenberg())
    assert engine_module.create_engine(matrix_graph, "simple", None) is None
    assert engine_module.create_engine(matrix_graph, "simple", None) is None
