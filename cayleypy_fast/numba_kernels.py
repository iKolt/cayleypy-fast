"""Optional numba tier (T4): fused CPU hash kernels for the fast engine.

Why this tier exists (measured, see docs/radical-speedup-plan.md "T4"):
torch's CPU int32 matmul path (``states.int32 @ vh``) does NOT dispatch to an
optimized BLAS integer GEMM, so the engine's per-step ``B x G`` neighbor-hash
matmul was measured at ~0.75-2 GMAC/s on a desktop CPU. A numba
``njit(parallel=True)`` kernel with the transposed ``(2G, n)`` layout and the
int8->int32 cast fused into the inner loop is 13-39x faster and bit-identical
(both int32 accumulation wrap mod 2^32 by C semantics; verified across dtypes
and graphs by ``tests/test_numba_tier.py`` against ``hasher.make_hashes``).

Numerical contract (bit-equality with the torch reference path):
  * state values are permutation indices (0 <= s < 2^31), so the int8->int32
    (or int64->int32) fusion is lossless;
  * every accumulate step wraps mod 2^32, matching torch int32 matmul;
  * ``out[:, 2g:2g+2]`` holds the two int32 hashes of generator ``g`` in the
    same order as ``PermutedHashVectors.vh`` columns, so the caller's
    ``.view(torch.int64)`` reinterpretation is unchanged.

The module is import-safe without numba: ``kernel_dual_int32`` stays ``None``
and the engine falls back to the torch matmul reference path. Kill-switch for
debugging: ``CAYLEYPY_FAST_NUMBA_DISABLE=1``.
"""

import os
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from .engine import PermutedHashVectors

_ENV_DISABLE = "CAYLEYPY_FAST_NUMBA_DISABLE"
_DISABLED_VALUES = {"1", "true", "yes", "on"}

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - numba is optional
    njit = None  # type: ignore[assignment,misc]
    prange = None  # type: ignore[assignment,misc]

# Module-level handle assigned conditionally below (see if-block).
# pylint: disable=invalid-name
_NUMBA_KERNEL_DUAL_I32 = None

if njit is not None:

    @njit(parallel=True, cache=True)
    # mypy: prange is numba's special iteration builtin, not visible to static analysis.
    def _hash_neighbors_dual_int32_np(states: np.ndarray, vh_t: np.ndarray, out: np.ndarray) -> None:
        """``out[i, c] = sum_j int32(states[i, j]) * vh_t[c, j]`` with mod-2^32 wraparound.

        States are read in their native dtype and each element is cast to int32
        first (lossless: permutation indices < 2^31), so the int32 multiply and
        int32 accumulate wrap exactly like torch's int32 matmul. ``vh_t`` is the
        transposed ``(2*G, n)`` int32 hash matrix so the inner loop reads both
        operands contiguously (SIMD-friendly).
        """
        rows = states.shape[0]
        cols = vh_t.shape[0]
        n = states.shape[1]
        # Numba types prange via its own typing pass; pylint/mypy see None and cannot infer iterability.
        for i in prange(rows):  # type: ignore[attr-defined]  # pylint: disable=not-an-iterable
            for c in range(cols):
                acc = np.int32(0)
                for j in range(n):
                    acc += np.int32(states[i, j]) * vh_t[c, j]
                out[i, c] = acc

    _NUMBA_KERNEL_DUAL_I32 = _hash_neighbors_dual_int32_np


def numba_available() -> bool:
    """Whether the numba tier can be used (imported and not env-disabled)."""
    if _NUMBA_KERNEL_DUAL_I32 is None:
        return False
    return os.environ.get(_ENV_DISABLE, "").strip().lower() not in _DISABLED_VALUES


def _numba_vh_t(hv: "PermutedHashVectors") -> np.ndarray:
    """Transposed contiguous ``(2*G, n)`` int32 copy of ``hv.vh`` (cached on the instance)."""
    cached = getattr(hv, "_numba_vh_t", None)
    if cached is not None:
        return cached
    vh_t = np.ascontiguousarray(hv.vh.numpy().T)
    setattr(hv, "_numba_vh_t", vh_t)
    return vh_t


def hash_neighbors_dual_int32_numba(hv: "PermutedHashVectors", states: torch.Tensor) -> torch.Tensor:
    """Numba dual-int32 hash of all ``B x G`` neighbors; bit-equal to the torch path.

    :param hv: Dual-int32 permuted hash vectors (``(n, 2*G)`` int32).
    :param states: ``(B, n)`` CPU states (int8/int64 small integer values; the
        kernel reads them directly, so no int32 cast temp is materialized).
    :return: ``(B, G)`` int64 hashes (torch interpretation of the int32 pairs).
    """
    assert hv.dual_int32, "numba kernel handles the dual-int32 path only"
    rows = states.shape[0]
    vh_t = _numba_vh_t(hv)
    cols = vh_t.shape[0]
    s_np = np.ascontiguousarray(states.numpy())
    out = np.empty((rows, cols), dtype=np.int32)
    assert _NUMBA_KERNEL_DUAL_I32 is not None
    _NUMBA_KERNEL_DUAL_I32(s_np, vh_t, out)
    h32 = torch.from_numpy(out)
    n_generators = cols // 2
    return h32.view(-1, n_generators, 2).view(torch.int64).reshape(-1, n_generators)


def pick_hash_fn(hv: "PermutedHashVectors") -> Optional[Callable[["PermutedHashVectors", torch.Tensor], torch.Tensor]]:
    """Return the numba dual-int32 hash fn if usable for these hash vectors, else None."""
    if hv.dual_int32 and numba_available():
        return hash_neighbors_dual_int32_numba
    return None
