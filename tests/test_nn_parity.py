"""T2b NN parity tests: legacy vs engine with learned (MLP) predictors.

The streaming NN scoring path (``engine.py`` ``_make_score_fn`` hybrid branch)
is exercised by the engine parity suite only for outcome parity
(``test_engine_beam_stays_capped_with_nn_predictor``). This file hardens NN
coverage (radical-plan T5, parity+smoke scope):

  * a **trained-in-test tiny MLP** on lrx8 (mirrors the upstream pretraining
    idiom at miniature scale: SGD on ``(state, distance)`` pairs from
    ``graph.random_walks``, fixed seeds), and
  * an **untrained seeded MLP** on cube222 (second predictor shape: 24-element
    state, multi-class one-hot),

with per-mode asserts in decreasing strength: ``path_found`` equality,
``path_length`` equality, ``validate_path`` on found paths, ``debug_scores``
min-score-sequence comparison (exact on CPU, ``allclose`` on CUDA), and — where
legal — scored-candidate parity observed through a recording model wrapper.

Why score-level asserts are mode-dependent: fp scores are deterministic on CPU,
but legacy ``search_iterated`` dedups later generators against the *post-topk
survivors* of earlier generators while the engine dedups against *all*
candidates of the step. Scored candidate sets (and hence ``debug_scores``) can
therefore legitimately diverge in ``iterated`` mode — measured empirically
(177 vs 149 scored rows on the fixed instance below) — so that mode keeps
outcome-level asserts only. On CUDA the MLP reduction order is not
batch-invariant, so fp bit-equality of scores is not a legal assert either;
outcome parity + ``allclose`` debug scores only.
"""

import pytest
import torch

from cayleypy import CayleyGraph, PermutationGroups, prepare_graph
from cayleypy.models.models import MlpModel, ModelConfig

import cayleypy_fast
from cayleypy_fast.engine import create_engine

_TRAINING_SEED = 20240


@pytest.fixture(autouse=True)
def _force_engine_on(monkeypatch):
    """Exercise the engine even on small beams (mirrors tests/test_engine_parity.py)."""
    monkeypatch.setenv("CAYLEYPY_FAST_MIN_BG", "0")
    monkeypatch.setenv("CAYLEYPY_FAST_MIN_BEAM", "0")


class _RecordingModel(torch.nn.Module):
    """Wraps a predictor model and records every scored input batch (detached CPU clones)."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.records: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.records.append(x.detach().to("cpu").clone())
        return self.inner(x)

    def clear(self) -> None:
        self.records.clear()


def _train_tiny_mlp_lrx8(device: str = "cpu"):
    """Build lrx8 graph + a tiny MLP trained on random-walk distances (deterministic).

    Construction order matters for determinism: the hasher self-seeds the global
    torch RNG at graph construction (``StateHasher.__init__``), random walks then
    consume that known state, and the model init is explicitly re-seeded.
    """
    graph = CayleyGraph(PermutationGroups.lrx(8), random_seed=42, device=device)
    walk_states, walk_distances = graph.random_walks(width=256, length=16)
    x, y = walk_states, walk_distances.float()
    torch.manual_seed(_TRAINING_SEED)
    model = MlpModel(ModelConfig(model_type="MLP", input_size=8, num_classes_for_one_hot=8, layers_sizes=[64, 64]))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(150):
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    model.eval()
    return graph, model


def _seeded_mlp_for(graph: CayleyGraph) -> torch.nn.Module:
    """Deterministic untrained MLP matching the graph's state encoding (fixed seed)."""
    max_state_value = int(graph.central_state.max())
    torch.manual_seed(_TRAINING_SEED)
    model = MlpModel(
        ModelConfig(
            model_type="MLP",
            input_size=graph.definition.state_size,
            num_classes_for_one_hot=max_state_value + 1,
            layers_sizes=[64, 64],
        )
    )
    return model.to(graph.device)


def _run_pair(graph, start_state, recorder, beam_mode: str, beam_width: int, history_depth: int):
    """Legacy then engine run with the same recording model; returns (results, records)."""
    kwargs = dict(
        beam_mode=beam_mode,
        predictor=recorder,
        beam_width=beam_width,
        max_steps=30,
        history_depth=history_depth,
        return_path=True,
    )
    recorder.clear()
    legacy_result = graph.beam_search(start_state=start_state, **kwargs)
    legacy_records = list(recorder.records)
    assert cayleypy_fast.enable()
    try:
        assert create_engine(graph, beam_mode, recorder) is not None, f"engine unavailable for {beam_mode}"
        recorder.clear()
        engine_result = graph.beam_search(start_state=start_state, **kwargs)
        engine_records = list(recorder.records)
    finally:
        cayleypy_fast.disable()
    return (legacy_result, engine_result), (legacy_records, engine_records)


def _assert_outcome_parity(graph, start_state, legacy_result, engine_result) -> None:
    """The universal, mode-independent asserts (legal on CPU and CUDA)."""
    assert engine_result.path_found == legacy_result.path_found
    assert engine_result.path_found, "fixed test instances must be solvable at these settings"
    assert engine_result.path_length == legacy_result.path_length
    graph.validate_path(start_state, engine_result.path)
    graph.validate_path(start_state, legacy_result.path)


def _assert_score_parity_exact(legacy_result, engine_result, legacy_records, engine_records) -> None:
    """CPU-only exact fp parity: debug score sequences + scored candidate multisets."""
    assert engine_result.debug_scores == legacy_result.debug_scores
    legacy_all = torch.cat(legacy_records) if legacy_records else torch.empty(0, 0)
    engine_all = torch.cat(engine_records) if engine_records else torch.empty(0, 0)
    # Same scored states as a multiset (engine streams fixed-size batches, legacy
    # scores per chunk — batch boundaries legitimately differ).
    assert sorted(map(tuple, engine_all.tolist())) == sorted(map(tuple, legacy_all.tolist()))


def _assert_score_parity_allclose(legacy_result, engine_result) -> None:
    """CUDA fp parity: identical steps' min scores match within float tolerance."""
    for step, legacy_score in legacy_result.debug_scores.items():
        assert step in engine_result.debug_scores
        assert legacy_score == pytest.approx(engine_result.debug_scores[step], rel=1e-5, abs=1e-5)


_MODES_HD = [("simple", 0), ("advanced", 2), ("iterated", 2), ("iterated_batched", 2)]
# Iterated keeps outcome-level asserts only (see module docstring: post-topk dedup divergence).
_EXACT_SCORE_MODES = {"simple", "advanced", "iterated_batched"}


@pytest.mark.parametrize("beam_mode,history_depth", _MODES_HD)
def test_nn_parity_trained_mlp_lrx8(beam_mode, history_depth):
    """Trained tiny MLP on lrx8: legacy vs engine parity on CPU (exact scores where legal)."""
    graph, model = _train_tiny_mlp_lrx8()
    recorder = _RecordingModel(model)
    start_state = graph.random_walks(width=1, length=41)[0][-1]

    (legacy_result, engine_result), (legacy_records, engine_records) = _run_pair(
        graph, start_state, recorder, beam_mode, beam_width=16, history_depth=history_depth
    )
    _assert_outcome_parity(graph, start_state, legacy_result, engine_result)
    if beam_mode in _EXACT_SCORE_MODES:
        _assert_score_parity_exact(legacy_result, engine_result, legacy_records, engine_records)
    # Recorded evidence that the streaming NN scoring path actually ran.
    assert len(engine_records) > 0


@pytest.mark.parametrize("beam_mode,history_depth", _MODES_HD)
def test_nn_parity_untrained_mlp_cube222(beam_mode, history_depth):
    """Untrained seeded MLP on cube222: a second predictor shape (multi-class one-hot)."""
    # device="cpu" is NOT optional: on GPU runners device="auto" would silently
    # pick CUDA, and the exact-fp asserts below are legal on CPU only.
    graph = CayleyGraph(prepare_graph("cube_2/2/2_6gensQTM"), random_seed=42, device="cpu")
    recorder = _RecordingModel(_seeded_mlp_for(graph))
    # length-17 walk at bw=1024: all 4 modes find (len 6) AFTER scoring begins,
    # so the streaming NN path is exercised on the found path (shorter walks
    # either simplify to distance <= 4 before any scoring, or never solve).
    start_state = graph.random_walks(width=1, length=17)[0][-1]

    (legacy_result, engine_result), (legacy_records, engine_records) = _run_pair(
        graph, start_state, recorder, beam_mode, beam_width=1024, history_depth=history_depth
    )
    _assert_outcome_parity(graph, start_state, legacy_result, engine_result)
    if beam_mode in _EXACT_SCORE_MODES:
        _assert_score_parity_exact(legacy_result, engine_result, legacy_records, engine_records)
    assert len(engine_records) > 0


# -----------------------------------------------------------------------------
# CUDA mirrors: same asserts minus fp bit-equality (see module docstring).
# Collected-but-skipped locally; executed on the Kaggle perf kernel (stage 0).
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only parity mirror")
@pytest.mark.parametrize("beam_mode,history_depth", _MODES_HD)
def test_nn_parity_trained_mlp_lrx8_cuda(beam_mode, history_depth):
    graph, model = _train_tiny_mlp_lrx8(device="cuda")
    recorder = _RecordingModel(model)
    start_state = graph.random_walks(width=1, length=41)[0][-1]

    (legacy_result, engine_result), (_, engine_records) = _run_pair(
        graph, start_state, recorder, beam_mode, beam_width=16, history_depth=history_depth
    )
    _assert_outcome_parity(graph, start_state, legacy_result, engine_result)
    _assert_score_parity_allclose(legacy_result, engine_result)
    assert len(engine_records) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only parity mirror")
@pytest.mark.parametrize("beam_mode,history_depth", _MODES_HD)
def test_nn_parity_untrained_mlp_cube222_cuda(beam_mode, history_depth):
    graph = CayleyGraph(prepare_graph("cube_2/2/2_6gensQTM"), random_seed=42, device="cuda")
    recorder = _RecordingModel(_seeded_mlp_for(graph))
    # See the CPU variant for why length 17 / bw 1024.
    start_state = graph.random_walks(width=1, length=17)[0][-1]

    (legacy_result, engine_result), (_, engine_records) = _run_pair(
        graph, start_state, recorder, beam_mode, beam_width=1024, history_depth=history_depth
    )
    _assert_outcome_parity(graph, start_state, legacy_result, engine_result)
    _assert_score_parity_allclose(legacy_result, engine_result)
    assert len(engine_records) > 0
