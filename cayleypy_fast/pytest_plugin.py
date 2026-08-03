"""pytest plugin: enable the fast engine in-process for a host test suite.

Usage (e.g. the upstream cayleypy suite with the engine active):

    pytest -p cayleypy_fast.pytest_plugin cayleypy/algo/beam_search_test.py

The plugin calls :func:`cayleypy_fast.enable` once at ``pytest_configure`` time
(after conftest collection), so every test in the process runs against the
patched ``BeamSearchAlgorithm``. Production size-gate thresholds apply unless
overridden via ``CAYLEYPY_FAST_MIN_BG`` / ``CAYLEYPY_FAST_MIN_BEAM``.
``CAYLEYPY_FAST_DISABLE=1`` turns the plugin into a no-op.
"""

from typing import Any

import cayleypy_fast


def pytest_configure(config: Any) -> None:  # pylint: disable=unused-argument
    """Enable the in-place patch for the duration of the pytest run."""
    cayleypy_fast.enable()
