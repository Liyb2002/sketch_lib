#!/usr/bin/env python3
"""
constraints_optimization/color_value_loss.py

Color/value loss based on the *heatmap point colors* removed by shrinking.

Goal (your spec):
- Loss = sum of "redness value" of points that were cut away.
- Redder  -> closer to 1
- Greener -> closer to 0
- Black   -> 0
- Empty (no points) -> 0

We keep the same public function names used elsewhere:
- load_heat_ply_points_and_heat
- to_local
- value_inside_bounds
- value_loss_0_1

Interpretation:
- load_heat_ply_points_and_heat returns:
    pts_world: (N,3)
    heat:      (N,) in [0,1] derived from point colors
- value_inside_bounds(...) returns:
    sum(heat) for points that lie inside [bmin,bmax] in *local coords*
- value_loss_0_1(items, sum_value0) returns:
    deleted_heat / sum_value0  (clipped to [0,1]),
  where deleted_heat is the total heat removed by shrinking:
    deleted_heat_i = max(0, value0_i - value_i)
"""

from typing import Any, Dict, List, Tuple
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# -----------------------------------------------------------------------------
# IO: load points + convert RGB -> "heat" in [0,1]
# -----------------------------------------------------------------------------

def _rgb_to_heat(rgb: np.ndarray) -> np.ndarray:
    """
    Map RGB -> redness score in [0,1].

    We use: heat = r / (r + g + b + eps)

    Properties:
    - pure red   (1,0,0) -> 1
    - pure green (0,1,0) -> 0
    - black      (0,0,0) -> 0 (handled explicitly)
    - yellow     (1,1,0) -> 0.5
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        return np.zeros((0,), dtype=np.float32)

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    s = r + g + b

    heat = np.zeros((rgb.shape[0],), dtype=np.float32)
    non_black = s > 1e-12
    heat[non_black] = (r[non_black] / (s[non_black] + 1e-12)).astype(np.float32)

    # clamp safety
    heat = np.clip(heat, 0.0, 1.0).astype(np.float32)
    return heat


def load_heat_ply_points_and_heat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a .ply point cloud where colors encode the heatmap.

    Returns:
      pts_world: (N,3) float64
      heat:      (N,)  float32 in [0,1]

    If open3d is unavailable or file has no points, returns empty arrays.
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
        heat = _rgb_to_heat(cols)

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
# Core: sum of heat inside bounds, and deleted-heat loss
# -----------------------------------------------------------------------------

def _inside_mask(local_pts: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    local_pts = np.asarray(local_pts, dtype=np.float64)
    bmin = np.asarray(bmin, dtype=np.float64).reshape(1, 3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(1, 3)
    if local_pts.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (local_pts[:, 0] >= bmin[0, 0]) & (local_pts[:, 0] <= bmax[0, 0]) &
        (local_pts[:, 1] >= bmin[0, 1]) & (local_pts[:, 1] <= bmax[0, 1]) &
        (local_pts[:, 2] >= bmin[0, 2]) & (local_pts[:, 2] <= bmax[0, 2])
    )


def value_inside_bounds(local_pts0: np.ndarray, heat_pow: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    """
    Returns the sum of heat values inside [bmin,bmax] in local coordinates.

    Notes:
    - heat_pow can be raw heat, or heat**gamma depending on caller.
    - If there are no points, returns 0.
    """
    lp = np.asarray(local_pts0, dtype=np.float64)
    h = np.asarray(heat_pow, dtype=np.float32).reshape(-1)
    if lp.size == 0 or h.size == 0:
        return 0.0
    if lp.shape[0] != h.shape[0]:
        # safety: mismatched shapes -> treat as empty
        return 0.0

    m = _inside_mask(lp, bmin, bmax)
    if m.size == 0:
        return 0.0
    return float(np.sum(h[m], dtype=np.float64))


def value_loss_0_1(items: List[Dict[str, Any]], sum_value0: float) -> float:
    """
    Deleted-heat loss in [0,1].

    Each item expects:
      value0 = sum heat inside original bounds (bmin0/bmax0)
      value  = sum heat inside current bounds  (bmin/bmax)

    deleted_heat = sum_i max(0, value0 - value)
    normalize by sum_value0 (total original heat sum).
    """
    denom = max(1e-12, float(sum_value0))
    if denom <= 1e-12:
        return 0.0

    deleted = 0.0
    for it in items:
        v0 = float(it.get("value0", 0.0))
        v = float(it.get("value", 0.0))
        deleted += max(0.0, v0 - v)

    return float(np.clip(deleted / denom, 0.0, 1.0))
