"""In-place activation machinery (guarded monkeypatch).

``enable()`` replaces the ``BeamSearchAlgorithm`` symbol in every cayleypy
module that exposes it with :class:`FastBeamSearchAlgorithm`, after running the
compatibility probe. ``disable()`` restores the originals. The environment
variable ``CAYLEYPY_FAST_DISABLE=1`` forces ``enable()`` to be a no-op.
"""

import importlib
import os
import warnings
from typing import TYPE_CHECKING, Any

from cayleypy.algo.beam_search import BeamSearchAlgorithm as _LegacyBeamSearchAlgorithm

from ._probe import PATCH_POINTS

if TYPE_CHECKING:
    from cayleypy import CayleyGraph
    from cayleypy.algo.beam_search_result import BeamSearchResult

ENV_DISABLE = "CAYLEYPY_FAST_DISABLE"
_DISABLED_VALUES = {"1", "true", "yes", "on"}


class FastBeamSearchAlgorithm(_LegacyBeamSearchAlgorithm):
    """Drop-in ``BeamSearchAlgorithm`` replacement.

    Each of the 4 ``search_*`` overrides routes to the fast engine when one is
    available for the (graph, mode, predictor) combination, and falls back to
    the legacy implementation (``super()``) otherwise. See the plan, section
    "Packaging & activation".
    """

    def _fast_engine(self, mode: str, predictor: Any) -> Any:
        try:
            from .engine import create_engine  # pylint: disable=import-outside-toplevel
        except ImportError:
            # Engine module drifted against this cayleypy version; stay on legacy.
            return None
        return create_engine(self.graph, mode, predictor)

    def search_simple(self, *args: Any, **kwargs: Any) -> "BeamSearchResult":
        engine = self._fast_engine("simple", kwargs.get("predictor"))
        if engine is not None:
            return engine.search_simple(*args, **kwargs)
        return super().search_simple(*args, **kwargs)

    def search_advanced(self, *args: Any, **kwargs: Any) -> "BeamSearchResult":
        engine = self._fast_engine("advanced", kwargs.get("predictor"))
        if engine is not None:
            return engine.search_advanced(*args, **kwargs)
        return super().search_advanced(*args, **kwargs)

    def search_iterated(self, *args: Any, **kwargs: Any) -> "BeamSearchResult":
        engine = self._fast_engine("iterated", kwargs.get("predictor"))
        if engine is not None:
            return engine.search_iterated(*args, **kwargs)
        return super().search_iterated(*args, **kwargs)

    def search_iterated_batched(self, *args: Any, **kwargs: Any) -> "BeamSearchResult":
        engine = self._fast_engine("iterated_batched", kwargs.get("predictor"))
        if engine is not None:
            return engine.search_iterated_batched(*args, **kwargs)
        return super().search_iterated_batched(*args, **kwargs)


class _PatchState:
    """Mutable module state for the patch (attribute mutation avoids `global` statements)."""

    def __init__(self) -> None:
        self.enabled = False
        self.originals: dict = {}


_STATE = _PatchState()


def _env_disabled() -> bool:
    return os.environ.get(ENV_DISABLE, "").strip().lower() in _DISABLED_VALUES


def enable() -> bool:
    """Patch cayleypy in-place so ``graph.beam_search(...)`` uses the fast engine.

    Runs the compatibility probe first; on any mismatch emits a ``RuntimeWarning``
    and leaves cayleypy untouched (legacy behaviour). Idempotent.

    :return: True if the patch is active after the call, False otherwise.
    """
    if _STATE.enabled:
        return True
    if _env_disabled():
        return False
    from ._probe import run_probe  # pylint: disable=import-outside-toplevel

    probe = run_probe()
    if not probe.ok:
        warnings.warn(
            "cayleypy_fast disabled: installed cayleypy is incompatible: " + "; ".join(probe.problems),
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    for module_name in PATCH_POINTS:
        module = importlib.import_module(module_name)
        _STATE.originals[module_name] = getattr(module, "BeamSearchAlgorithm")
        setattr(module, "BeamSearchAlgorithm", FastBeamSearchAlgorithm)
    _STATE.enabled = True
    return True


def disable() -> bool:
    """Restore the original ``BeamSearchAlgorithm`` symbols.

    :return: True if the patch was active and has been removed, False if it was not active.
    """
    if not _STATE.enabled:
        return False
    for module_name, original in _STATE.originals.items():
        setattr(importlib.import_module(module_name), "BeamSearchAlgorithm", original)
    _STATE.originals.clear()
    _STATE.enabled = False
    return True


def is_enabled() -> bool:
    """Whether the in-place patch is currently active."""
    return _STATE.enabled


class FastGraphWrapper:
    """Patch-averse wrapper exposing ``beam_search`` with the cayleypy signature.

    Uses :class:`FastBeamSearchAlgorithm` directly, so the fast engine is used
    (when available for the graph/mode/predictor) without monkeypatching
    cayleypy. All other attribute access is delegated to the wrapped graph.
    """

    def __init__(self, graph: "CayleyGraph") -> None:
        self._graph = graph

    @property
    def wrapped_graph(self) -> "CayleyGraph":
        """The underlying :class:`cayleypy.CayleyGraph`."""
        return self._graph

    def beam_search(self, **kwargs: Any) -> "BeamSearchResult":
        """Run beam search via the fast engine (same signature as ``CayleyGraph.beam_search``)."""
        return FastBeamSearchAlgorithm(self._graph).search(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)


def wrap(graph: "CayleyGraph") -> FastGraphWrapper:
    """Return a patch-averse wrapper around ``graph`` (see :class:`FastGraphWrapper`)."""
    return FastGraphWrapper(graph)
