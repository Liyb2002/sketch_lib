# AEP/connect_change.py
"""
AEP/connect_change.py

Implements "possible anchor point change" for a connected relation.

Anchor definition (matches what we used before):
- Given AABB(A) and AABB(B), compute closest points pA and pB (axis-wise),
  then anchor/pin = midpoint(pA, pB).

We compute:
- old_anchor using A before-edit AABB vs B
- new_anchor using A after-edit AABB vs B

If anchor moves more than eps (L2), we say neighbor needs edit.
"""

from typing import Any, Tuple
import numpy as np


def _np3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)


def _aabb_closest_points(
    mnA: np.ndarray, mxA: np.ndarray,
    mnB: np.ndarray, mxB: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Axis-wise closest points between two AABBs.
    If overlapping along an axis, pick midpoint of overlap interval on that axis.
    Returns (pA, pB, pin_midpoint).
    """
    pA = np.zeros((3,), dtype=np.float64)
    pB = np.zeros((3,), dtype=np.float64)

    for k in range(3):
        if mxA[k] < mnB[k]:
            # A is left of B
            pA[k] = mxA[k]
            pB[k] = mnB[k]
        elif mxB[k] < mnA[k]:
            # B is left of A
            pA[k] = mnA[k]
            pB[k] = mxB[k]
        else:
            # overlap along this axis: pick a common coordinate
            lo = max(mnA[k], mnB[k])
            hi = min(mxA[k], mxB[k])
            mid = 0.5 * (lo + hi)
            pA[k] = mid
            pB[k] = mid

    pin = 0.5 * (pA + pB)
    return pA, pB, pin


def compute_anchor_change(
    mnA_before: np.ndarray,
    mxA_before: np.ndarray,
    mnA_after: np.ndarray,
    mxA_after: np.ndarray,
    mnB: np.ndarray,
    mxB: np.ndarray,
    eps: float,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Returns:
      (changed, old_pin, new_pin)

    changed if ||new_pin - old_pin||_2 > eps
    """
    mnA_before = _np3(mnA_before)
    mxA_before = _np3(mxA_before)
    mnA_after  = _np3(mnA_after)
    mxA_after  = _np3(mxA_after)
    mnB = _np3(mnB)
    mxB = _np3(mxB)

    _, _, old_pin = _aabb_closest_points(mnA_before, mxA_before, mnB, mxB)
    _, _, new_pin = _aabb_closest_points(mnA_after,  mxA_after,  mnB, mxB)

    d = float(np.linalg.norm(new_pin - old_pin))
    changed = d > float(eps)

    return changed, old_pin, new_pin
