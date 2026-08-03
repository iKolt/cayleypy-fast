"""Deterministic seeding for the cayleypy-fast test suite (mirrors cayleypy's conftest).

An autouse fixture seeds ``numpy``, Python ``random`` and ``torch`` to a fixed
value before each test, so characterization/parity tests are reproducible.
Tests that need their own randomness can set a seed locally or use the
``deterministic_seed`` fixture.
"""

import random

import numpy as np
import pytest
import torch

# Fixed seed for the whole suite (same value as cayleypy's conftest).
DETERMINISTIC_SEED = 12345


@pytest.fixture(autouse=True)
def _seed_everything():
    """Seed numpy, Python ``random`` and torch before each test for determinism."""
    random.seed(DETERMINISTIC_SEED)
    np.random.seed(DETERMINISTIC_SEED)
    torch.manual_seed(DETERMINISTIC_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(DETERMINISTIC_SEED)
    yield


@pytest.fixture
def deterministic_seed() -> int:
    """The fixed seed used by the suite. Useful when a test needs to pass it on."""
    return DETERMINISTIC_SEED
