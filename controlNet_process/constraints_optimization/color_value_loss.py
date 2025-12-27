#!/usr/bin/env python3
"""
constraints_optimization/color_value_loss.py

Point-based value loss := weighted COUNT of points that get cut away.

Your spec:
- Each point has a scalar "value" = (redness) ** 8
- Red   -> value near 1
- Green -> value near 0
- Black -> 0
- Empty (no point) -> 0
- Cutting more points always increases penalty via SUM of point values removed.

We keep the same function names used by optimizer.py:
- load_heat_ply_points_and_heat
- to_local
- value_inside_bounds
- value_loss_0_1

Definition:
- load_heat_ply_points_and_heat returns:
    pts_world: (N,3)
    heat:      (N,) where heat = (redness ** 8) in [0,1]
- value_inside_bounds(...) returns:
    sum(heat) for points inside [bmin,bmax] in local coords
- value_loss_0_1(items, sum_value0) returns:
    removed_value / sum_value0  (clipped to [0,1]),
  where removed_value = sum_i max(0, value0_i - value_i)

Notes:
- This is SUM, not average.
- Normalization by sum_value0 keeps the loss in [0,1] and scale-stable across scenes.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# -----------------------------------------------------------------------------
# Color -> redness -> heat = redness^8
# -----------------------------------------------------------------------------

def _rgb_to_redness(rgb: np.ndarray) -> np.ndarray:
    """
    Map RGB -> redness score in [0,1].

    redness = r / (r + g + b + eps)

    - pure red   (1,0,0) -> 1
    - pure green (0,1,0) -> 0
    - black      (0,0,0) -> 0
    - yellow     (1,1,0) -> 0.5
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        return np.zeros((0,), dtype=np.float32)

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    s = r + g + b

    red = np.zeros((rgb.shape[0],), dtype=np.float32)
    m = s > 1e-12
    red[m] = r[m] / (s[m] + 1e-12)
    return np.clip(red, 0.0, 1.0).astype(np.float32)


def _redness_to_value(redness: np.ndarray, power: int = 8) -> np.ndarray:
    """
    point_value = redness ** power
    """
    red = np.asarray(redness, dtype=np.float32).reshape(-1)
    # Ensure non-negative, then power
    red = np.clip(red, 0.0, 1.0)
    return np.power(red, float(power), dtype=np.float32)


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def load_heat_ply_points_and_heat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a .ply point cloud where colors encode the heatmap.

    Returns:
      pts_world: (N,3) float64
      heat:      (N,)  float32, where heat = (redness ** 8) in [0,1]

    If open3d is unavailable, returns empty arrays.
    If the PLY has no colors, heat is zeros.
    """
    if o3d is None:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float32)

    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float32)

    cols = np.asarray(pcd.colors, dtype=np.float32) if len(pcd.colors) == len(pcd.points) else None
    if cols is None or cols.size == 0:
        heat = np.zeros((pts.shape[0],), dtype=np.float32)
    else:
        # open3d colors are typically already in [0,1]
        red = _rgb_to_redness(cols)
        heat = _redness_to_value(red, power=8)

    return pts, heat


# -----------------------------------------------------------------------------
# Geometry: world -> local
# -----------------------------------------------------------------------------

def to_local(points_world: np.ndarray, center0_world: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Transform world points into the (center0_world, R) local frame:
      local = (world - center0) @ R^T
    """
    pw = np.asarray(points_world, dtype=np.float64)
    c = np.asarray(center0_world, dtype=np.float64).reshape(1, 3)
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return (pw - c) @ Rm.T


# -----------------------------------------------------------------------------
# Core: sum of point values inside bounds, and removed-value loss
# -----------------------------------------------------------------------------

def _inside_mask(local_pts: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    lp = np.asarray(local_pts, dtype=np.float64)
    bmin = np.asarray(bmin, dtype=np.float64).reshape(1, 3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(1, 3)
    if lp.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (lp[:, 0] >= bmin[0, 0]) & (lp[:, 0] <= bmax[0, 0]) &
        (lp[:, 1] >= bmin[0, 1]) & (lp[:, 1] <= bmax[0, 1]) &
        (lp[:, 2] >= bmin[0, 2]) & (lp[:, 2] <= bmax[0, 2])
    )


def value_inside_bounds(local_pts0: np.ndarray, heat_pow: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    """
    Backward-compat signature.

    Returns:
      sum(point_value) for points inside [bmin,bmax] in local coordinates.

    heat_pow is assumed to already be the per-point value (redness^8),
    OR (redness^(8*gamma)) if the caller applied an extra gamma. Either way,
    we just sum it.
    """
    lp = np.asarray(local_pts0, dtype=np.float64)
    h = np.asarray(heat_pow, dtype=np.float32).reshape(-1)
    if lp.size == 0 or h.size == 0:
        return 0.0
    if lp.shape[0] != h.shape[0]:
        return 0.0

    m = _inside_mask(lp, bmin, bmax)
    if m.size == 0:
        return 0.0
    return float(np.sum(h[m], dtype=np.float64))


def value_loss_0_1(items: List[Dict[str, Any]], sum_value0: float) -> float:
    """
    Global removed-value loss in [0,1].

    Each item:
      value0 = sum(point_value) inside original bounds
      value  = sum(point_value) inside current bounds
      removed_i = max(0, value0 - value)

    Normalize by sum_value0 (total original value).
    """
    denom = max(1e-12, float(sum_value0))
    if denom <= 1e-12:
        return 0.0

    removed = 0.0
    for it in items:
        v0 = float(it.get("value0", 0.0))
        v = float(it.get("value", 0.0))
        removed += max(0.0, v0 - v)

    return float(np.clip(removed / denom, 0.0, 1.0))
