#!/usr/bin/env python3
"""
constraints_optimization/color_value_loss.py

Per-label point "value" from PLY colors.

Spec:
- Redder => closer to 1, greener => close to 0, black => 0
- Point value = (redness) ** 8
  redness = r / (r+g+b+eps)

Returns:
- pts_world: (N,3)
- heat: (N,) in [0,1] where black contributes 0
"""

from typing import Tuple
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


def _rgb_to_redness(rgb: np.ndarray) -> np.ndarray:
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
    red = np.asarray(redness, dtype=np.float32).reshape(-1)
    red = np.clip(red, 0.0, 1.0)
    heat = np.power(red, float(power)).astype(np.float32)
    # hard-zero tiny values (avoid near-black noise)
    heat[heat < 1e-6] = 0.0
    return heat


def load_heat_ply_points_and_heat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pts_world: (N,3) float64
      heat:      (N,)  float32 value = redness^8, black -> 0
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
        red = _rgb_to_redness(cols)
        heat = _redness_to_value(red, power=8)

    return pts, heat
