#!/usr/bin/env python3
"""
constraints_optimization/color_value_loss.py

Heat decode + value-in-box + normalized color/value loss in [0,1].

Contains:
- heat_from_red_green_black (matches heat_map.py colormap)
- load_heat_ply_points_and_heat
- to_local
- value_inside_bounds
- value_loss_0_1
"""

from typing import Any, Dict, List, Tuple
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# -----------------------------------------------------------------------------
# Heat decode (matches heat_map.py colormap)
# -----------------------------------------------------------------------------

def heat_from_red_green_black(colors_0_1: np.ndarray, power: float = 4.0) -> np.ndarray:
    """
    Reverse heat_map.py colormap:
      h<=0.5 : rgb=(0, 2h, 0)         => h=0.5*g
      h>=0.5 : rgb=(2h-1, 2-2h, 0)    => h=0.5+0.5*r
    """
    c = np.asarray(colors_0_1, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] < 2:
        return np.zeros((c.shape[0],), dtype=np.float32)
    r = c[:, 0]
    g = c[:, 1]
    h = np.where(r > 1e-6, 0.5 + 0.5 * r, 0.5 * g)
    h = np.clip(h, 0.0, 1.0)
    h = h ** float(power)   # steeper, no cutoff
    return h


# -----------------------------------------------------------------------------
# IO for heat PLY
# -----------------------------------------------------------------------------

def load_heat_ply_points_and_heat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if o3d is None:
        raise RuntimeError("open3d required to read PLY. pip install open3d")
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts, np.zeros((0,), dtype=np.float32)
    heat = heat_from_red_green_black(cols)
    return pts, heat


# -----------------------------------------------------------------------------
# Geometry: WORLD -> LOCAL (anchored at original center0_world with axes R)
# -----------------------------------------------------------------------------

def to_local(points_world: np.ndarray, center0_world: np.ndarray, R: np.ndarray) -> np.ndarray:
    c = np.asarray(center0_world, dtype=np.float64).reshape(1, 3)
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return (np.asarray(points_world, dtype=np.float64) - c) @ Rm.T


# -----------------------------------------------------------------------------
# Value inside bounds and normalized loss
# -----------------------------------------------------------------------------

def value_inside_bounds(local_pts0: np.ndarray, heat_pow: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    bmin = np.asarray(bmin, dtype=np.float64).reshape(1, 3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(1, 3)
    inside = np.all((local_pts0 >= bmin) & (local_pts0 <= bmax), axis=1)
    return float(np.sum(np.asarray(heat_pow, dtype=np.float64)[inside]))


def value_loss_0_1(items: List[Dict[str, Any]], sum_value0: float) -> float:
    """
    Uses the same normalization you already had:
      L = clip( lost / (0.1 * sum_value0), 0, 1)
    where lost = Σ (value0 - value)
    """
    lost = 0.0
    for it in items:
        lost += float(it["value0"] - it["value"])
    return float(np.clip(lost / max(1e-12, float(sum_value0 * 0.1)), 0.0, 1.0))
