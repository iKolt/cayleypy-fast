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
