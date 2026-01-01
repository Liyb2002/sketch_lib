#!/usr/bin/env python3
"""
AEP/same_pair_neighbor_edits.py

Resize-only propagation for same_pair components.

Goal:
- same_pair means two components should undergo the same *size change* (not translation).
- We keep B's center fixed.
- Because A and B may be oriented differently, we find an axis permutation that best matches
  A_before extents to B_before extents (orientation-invariant matching).
- Then we apply A's per-axis resize ratios to B along the matched axes.

Returned:
- mnB_after, mxB_after: resized AABB for B
- size_deltaB: (size_after - size_before) in B world axes
- axis_map: tuple of length 3 describing mapping (B_axis -> A_axis)
"""

from typing import Tuple
import numpy as np


def _sizes(mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return np.maximum(mx - mn, 0.0)


def _center(mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return 0.5 * (mn + mx)


def _all_axis_maps() -> Tuple[Tuple[int, int, int], ...]:
    # axis_map is (B_x -> A_axis, B_y -> A_axis, B_z -> A_axis)
    return (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )


def _best_axis_map(sizeA_before: np.ndarray, sizeB_before: np.ndarray, eps: float) -> Tuple[int, int, int]:
    """
    Pick axis_map (B_axis -> A_axis) that makes sizeA_before[axis_map] closest to sizeB_before.
    """
    best_map = (0, 1, 2)
    best_cost = float("inf")

    # Normalize to reduce scale sensitivity
    denom = np.maximum(np.linalg.norm(sizeB_before), eps)

    for m in _all_axis_maps():
        predB = sizeA_before[list(m)]
        cost = float(np.linalg.norm(predB - sizeB_before) / denom)
        if cost < best_cost:
            best_cost = cost
            best_map = m

    return best_map


def resize_same_pair_neighbor(
    *,
    mnA_before: np.ndarray,
    mxA_before: np.ndarray,
    mnA_after: np.ndarray,
    mxA_after: np.ndarray,
    mnB: np.ndarray,
    mxB: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """
    Apply A's resize (from A_before->A_after) to B, but only resizing (no translation),
    with axis permutation matching to handle different orientations.

    - Compute A resize ratios per axis: rA = sizeA_after / max(sizeA_before, eps)
    - Find axis_map (B_axis -> A_axis) by matching sizeA_before (permuted) to sizeB_before
    - Apply ratios to B: sizeB_after[j] = sizeB_before[j] * rA[ axis_map[j] ]
    - Keep B center fixed
    """
    mnA_before = np.asarray(mnA_before, dtype=np.float64).reshape(3)
    mxA_before = np.asarray(mxA_before, dtype=np.float64).reshape(3)
    mnA_after  = np.asarray(mnA_after,  dtype=np.float64).reshape(3)
    mxA_after  = np.asarray(mxA_after,  dtype=np.float64).reshape(3)
    mnB        = np.asarray(mnB,        dtype=np.float64).reshape(3)
    mxB        = np.asarray(mxB,        dtype=np.float64).reshape(3)

    sizeA_before = _sizes(mnA_before, mxA_before)
    sizeA_after  = _sizes(mnA_after,  mxA_after)
    sizeB_before = _sizes(mnB, mxB)

    # Per-axis resize ratios from A (avoid division by 0)
    denomA = np.maximum(sizeA_before, eps)
    rA = sizeA_after / denomA

    # Choose best axis permutation (orientation-invariant)
    axis_map = _best_axis_map(sizeA_before=sizeA_before, sizeB_before=sizeB_before, eps=eps)

    # Apply A's ratios to B along matched axes
    rB = rA[list(axis_map)]  # rB[j] corresponds to B_axis j
    sizeB_after = sizeB_before * rB

    # Clamp tiny/negative sizes
    sizeB_after = np.maximum(sizeB_after, eps)

    # Keep B center fixed
    cB = _center(mnB, mxB)
    half = 0.5 * sizeB_after
    mnB_after = cB - half
    mxB_after = cB + half

    size_deltaB = sizeB_after - sizeB_before
    return mnB_after, mxB_after, size_deltaB, axis_map
