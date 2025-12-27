#!/usr/bin/env python3
"""
constraints_optimization/color_value_loss.py

Simple "color/value" loss := amount of space deleted (volume removed), nothing else.

Interpretation:
- Each box has an original local extent extent0 (from input OBB).
- Current box volume comes from current local bounds (bmin/bmax).
- Deleted volume = max(0, vol0 - vol).
- Global loss is normalized to [0,1] by total original volume (with a 0.1 factor to
  keep behavior similar to the old "budget" style scaling).

Keeps the same function names used by optimizer.py:
- load_heat_ply_points_and_heat (kept for backward-compat; not used)
- to_local (kept; not used)
- value_inside_bounds (kept; returns volume, but optimizer uses it only to fill it["value"])
- value_loss_0_1 (computes normalized deleted-volume loss)
"""

from typing import Any, Dict, List, Tuple
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# -----------------------------------------------------------------------------
# Backward-compat stubs (optimizer imports these)
# -----------------------------------------------------------------------------

def load_heat_ply_points_and_heat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compat: optimizer still calls this. We return points + dummy zeros.
    If open3d is unavailable, return empty arrays (safe because we don't use them).
    """
    if o3d is None:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float32)
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    # second return used to be heat; keep shape consistent
    return pts, np.zeros((pts.shape[0],), dtype=np.float32)


def to_local(points_world: np.ndarray, center0_world: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Backward-compat: return a local transform (not used for volume-based loss).
    """
    c = np.asarray(center0_world, dtype=np.float64).reshape(1, 3)
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return (np.asarray(points_world, dtype=np.float64) - c) @ Rm.T


# -----------------------------------------------------------------------------
# Volume-based "value" proxy
# -----------------------------------------------------------------------------

def _volume_from_bounds(bmin: np.ndarray, bmax: np.ndarray) -> float:
    bmin = np.asarray(bmin, dtype=np.float64).reshape(3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(3)
    ext = np.maximum(bmax - bmin, 0.0)
    return float(ext[0] * ext[1] * ext[2])


def value_inside_bounds(local_pts0: np.ndarray, heat_pow: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    """
    Backward-compat signature.

    In the new definition, "value" is just the CURRENT VOLUME of the box
    implied by (bmin,bmax), independent of points/colors.

    optimizer.py stores:
      it["value0"] = value_inside_bounds(..., bmin0, bmax0)   -> original volume
      it["value"]  = value_inside_bounds(..., bmin,  bmax)    -> current volume
    """
    return _volume_from_bounds(bmin, bmax)


def value_loss_0_1(items: List[Dict[str, Any]], sum_value0: float) -> float:
    """
    Global deleted-volume loss in [0,1].

    Each item:
      value0 = original volume
      value  = current volume
      deleted = max(0, value0 - value)

    Normalize by total original volume (sum_value0).

    NOTE: optimizer passes sum_value0 computed from it["value0"].
    """
    deleted = 0.0
    for it in items:
        v0 = float(it.get("value0", 0.0))
        v = float(it.get("value", 0.0))
        deleted += max(0.0, v0 - v)

    # Normalize: fraction of total original volume removed.
    # (No extra "budget" factor; pure geometry.)
    return float(np.clip(deleted / max(1e-12, float(sum_value0)), 0.0, 1.0))