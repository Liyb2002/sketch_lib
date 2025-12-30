# AEP/connect_neighbor_edits.py
"""
AEP/connect_neighbor_edits.py

Simple connected-neighbor edit: translation only.

Idea:
- Preserve relative location between A and B by translating B by the same
  translation applied to A's AABB center.

Given:
- A before/after AABB
- B current AABB (typically still "before")

Compute:
- cA_before, cA_after
- delta = cA_after - cA_before
Apply:
- mnB_new = mnB + delta
- mxB_new = mxB + delta

Return:
- (mnB_new, mxB_new, delta)
"""

from typing import Any, Tuple
import numpy as np


def _np3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)


def _aabb_center(mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return 0.5 * (_np3(mn) + _np3(mx))


def translate_neighbor_by_target_delta(
    mnA_before: np.ndarray,
    mxA_before: np.ndarray,
    mnA_after: np.ndarray,
    mxA_after: np.ndarray,
    mnB: np.ndarray,
    mxB: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Translation-only propagation for connected neighbor.

    Returns:
      mnB_new, mxB_new, delta
    """
    mnA_before = _np3(mnA_before)
    mxA_before = _np3(mxA_before)
    mnA_after  = _np3(mnA_after)
    mxA_after  = _np3(mxA_after)
    mnB = _np3(mnB)
    mxB = _np3(mxB)

    cA_before = _aabb_center(mnA_before, mxA_before)
    cA_after  = _aabb_center(mnA_after,  mxA_after)
    delta = cA_after - cA_before

    mnB_new = mnB + delta
    mxB_new = mxB + delta
    return mnB_new, mxB_new, delta
