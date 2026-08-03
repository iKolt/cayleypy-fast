"""Test-isolation fixture: never leak the in-place patch across tests."""

import pytest

import cayleypy_fast


@pytest.fixture(autouse=True)
def _no_patch_leak():
    """Ensure the patch is disabled before and after every test."""
    cayleypy_fast.disable()
    yield
    cayleypy_fast.disable()
