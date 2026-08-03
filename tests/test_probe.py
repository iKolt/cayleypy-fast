"""Tests for the cayleypy compatibility probe."""

import cayleypy.algo.beam_search as beam_search_module

import cayleypy_fast


def test_probe_passes_on_current_cayleypy():
    result = cayleypy_fast.run_probe()
    assert result.ok, f"probe problems: {result.problems}"
    assert result.problems == []


def test_probe_detects_missing_helper(monkeypatch):
    monkeypatch.delattr(beam_search_module, "_check_path_found")
    result = cayleypy_fast.run_probe()
    assert not result.ok
    assert any("_check_path_found" in problem for problem in result.problems)


def test_probe_detects_missing_hashset_method(monkeypatch):
    monkeypatch.delattr(beam_search_module.TorchHashSet, "get_merged_sorted")
    result = cayleypy_fast.run_probe()
    assert not result.ok
    assert any("TorchHashSet" in problem for problem in result.problems)


def test_probe_detects_search_signature_drift(monkeypatch):
    def narrowed_search(self, *, start_state):
        raise NotImplementedError

    monkeypatch.setattr(beam_search_module.BeamSearchAlgorithm, "search", narrowed_search)
    result = cayleypy_fast.run_probe()
    assert not result.ok
    assert any("search" in problem for problem in result.problems)


def test_probe_detects_pre_audit_restore_path(monkeypatch):
    """Pre-audit-fix cayleypy has ``_restore_path`` without ``destination_state`` — must be rejected."""

    def old_restore_path(found_layer_id, _new_hashes, _new_states, graph, restore_path_hashes, bfs_layers_hashes, bfs):
        raise NotImplementedError

    monkeypatch.setattr(beam_search_module, "_restore_path", old_restore_path)
    result = cayleypy_fast.run_probe()
    assert not result.ok
    assert any("_restore_path" in problem for problem in result.problems)
