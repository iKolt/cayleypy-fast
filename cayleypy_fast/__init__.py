"""cayleypy-fast: radically faster beam search for CayleyPy.

Usage (in-place activation via guarded monkeypatch):

    import cayleypy_fast
    cayleypy_fast.enable()   # All graph.beam_search(...) calls now use the fast engine when available.

Patch-averse alternative:

    wrapped = cayleypy_fast.wrap(graph)
    result = wrapped.beam_search(start_state=...)

Set the environment variable ``CAYLEYPY_FAST_DISABLE=1`` to force the legacy
implementation (``enable()`` becomes a no-op).

Design doc: docs/radical-speedup-plan.md.
"""

from ._patch import ENV_DISABLE, FastBeamSearchAlgorithm, FastGraphWrapper, disable, enable, is_enabled, wrap
from ._probe import ProbeResult, run_probe

__version__ = "0.1.0"

__all__ = [
    "ENV_DISABLE",
    "FastBeamSearchAlgorithm",
    "FastGraphWrapper",
    "ProbeResult",
    "disable",
    "enable",
    "is_enabled",
    "run_probe",
    "wrap",
]
