#!/usr/bin/env python3
"""
constraints_optimization/overlap_loss.py

WORLD-AABB overlap loss + basic geometry helpers needed by optimizer.

Contains:
- Local asymmetric OBB bounds -> WORLD AABB helpers (center0 + R + bmin/bmax)
- Pairwise WORLD-AABB overlap volume
- Normalized overlap loss in [0,1]
"""

from typing import Any, Dict, List, Tuple
import numpy as np


# -----------------------------------------------------------------------------
# Geometry (LOCAL bounds -> WORLD AABB), asymmetric bmin/bmax
# -----------------------------------------------------------------------------

def local_center_from_bounds(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    bmin = np.asarray(bmin, dtype=np.float64).reshape(3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(3)
    return 0.5 * (bmin + bmax)


def extent_from_bounds(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    bmin = np.asarray(bmin, dtype=np.float64).reshape(3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(3)
    return np.maximum(bmax - bmin, 0.0)


def center_world_from_bounds(center0_world: np.ndarray, R: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    """
    Given original world center0 and rotation R, plus *current* local bmin/bmax,
    compute the updated world center corresponding to the local center shift.
    """
    c0 = np.asarray(center0_world, dtype=np.float64).reshape(3)
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    lc = local_center_from_bounds(bmin, bmax).reshape(3)
    return c0 + (Rm @ lc)


def obb_corners_world_asym(center0_world: np.ndarray, R: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    c0 = np.asarray(center0_world, dtype=np.float64).reshape(3)
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    bmin = np.asarray(bmin, dtype=np.float64).reshape(3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(3)

    xs = [bmin[0], bmax[0]]
    ys = [bmin[1], bmax[1]]
    zs = [bmin[2], bmax[2]]

    corners_local = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    corners_world = (Rm @ corners_local.T).T + c0[None, :]
    return corners_world


def obb_world_aabb_asym(center0_world: np.ndarray, R: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    corners = obb_corners_world_asym(center0_world, R, bmin, bmax)
    return corners.min(axis=0), corners.max(axis=0)


# -----------------------------------------------------------------------------
# AABB overlap (WORLD)
# -----------------------------------------------------------------------------

def box_volume(mn: np.ndarray, mx: np.ndarray) -> float:
    ext = np.asarray(mx, dtype=np.float64) - np.asarray(mn, dtype=np.float64)
    ext = np.maximum(ext, 0.0)
    return float(ext[0] * ext[1] * ext[2])


def pairwise_overlap_volume(mn1: np.ndarray, mx1: np.ndarray, mn2: np.ndarray, mx2: np.ndarray) -> float:
    omax = np.minimum(mx1, mx2)
    omin = np.maximum(mn1, mn2)
    oext = np.maximum(omax - omin, 0.0)
    return float(oext[0] * oext[1] * oext[2])


def compute_pairwise_overlaps(mins: np.ndarray, maxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)
    n = int(mins.shape[0])

    vols = np.array([box_volume(mins[i], maxs[i]) for i in range(n)], dtype=np.float64)
    per_box = np.zeros((n,), dtype=np.float64)

    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            v = pairwise_overlap_volume(mins[i], maxs[i], mins[j], maxs[j])
            if v > 0.0:
                pairs += 1
                total += v
                per_box[i] += v
                per_box[j] += v
    return vols, per_box, float(total), int(pairs)


def overlap_loss_0_1(mins: np.ndarray, maxs: np.ndarray, eps: float = 1e-12) -> Tuple[float, float, float, int, np.ndarray, np.ndarray]:
    """
    Returns:
      L in [0,1],
      inter_sum,
      denom,
      overlap_pairs,
      per_box_overlap,
      vols
    """
    vols, per_box_overlap, inter_sum, overlap_pairs = compute_pairwise_overlaps(mins, maxs)

    denom = 0.0
    n = int(mins.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            denom += float(min(vols[i], vols[j]))

    L = float(inter_sum) / max(float(eps), float(denom))
    L = float(np.clip(L, 0.0, 1.0))
    return L, float(inter_sum), float(denom), int(overlap_pairs), per_box_overlap, vols
