#!/usr/bin/env python3
"""
constraints_optimization/no_overlap_loss.py

Loss utilities for asymmetric shrink-only bounding box optimization.

This file contains:
- Heat decoding from heat_map.py colors
- Local<->world geometry helpers for asymmetric bounds (bmin/bmax)
- WORLD-AABB overlap loss in [0,1]
- Heat^gamma value-in-box + normalized value loss in [0,1]
- Objective assembly: w_overlap * L_ov + w_value * L_val

No optimization loop lives here.
"""

import json
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Heat decode (matches heat_map.py colormap)
# -----------------------------------------------------------------------------

def heat_from_red_green_black(colors_0_1: np.ndarray) -> np.ndarray:
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
    return np.clip(h, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Geometry (LOCAL bounds -> WORLD AABB), asymmetric bmin/bmax
# -----------------------------------------------------------------------------

def to_local(points_world: np.ndarray, center0_world: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Local frame anchored at original center0_world with axes R.
    """
    c = np.asarray(center0_world, dtype=np.float64).reshape(1, 3)
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return (np.asarray(points_world, dtype=np.float64) - c) @ Rm.T


def local_center_from_bounds(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    bmin = np.asarray(bmin, dtype=np.float64).reshape(3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(3)
    return 0.5 * (bmin + bmax)


def extent_from_bounds(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    bmin = np.asarray(bmin, dtype=np.float64).reshape(3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(3)
    return np.maximum(bmax - bmin, 0.0)


def center_world_from_bounds(center0_world: np.ndarray, R: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
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
    vols, per_box_overlap, inter_sum, overlap_pairs = compute_pairwise_overlaps(mins, maxs)

    denom = 0.0
    n = int(mins.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            denom += float(min(vols[i], vols[j]))

    L = float(inter_sum) / max(float(eps), float(denom))
    L = float(np.clip(L, 0.0, 1.0))
    return L, float(inter_sum), float(denom), int(overlap_pairs), per_box_overlap, vols


# -----------------------------------------------------------------------------
# Value inside bounds (NONLINEAR heat^gamma)
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


def value_inside_bounds(local_pts0: np.ndarray, heat_pow: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    bmin = np.asarray(bmin, dtype=np.float64).reshape(1, 3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(1, 3)
    inside = np.all((local_pts0 >= bmin) & (local_pts0 <= bmax), axis=1)
    return float(np.sum(heat_pow[inside]))


def value_loss_0_1(items: List[Dict[str, Any]], sum_value0: float) -> float:
    lost = 0.0
    for it in items:
        lost += float(it["value0"] - it["value"])
    return float(np.clip(lost / max(1e-12, float(sum_value0)), 0.0, 1.0))


# -----------------------------------------------------------------------------
# Objective assembly
# -----------------------------------------------------------------------------

def objective_from_world_aabbs(
    *,
    mins: np.ndarray,
    maxs: np.ndarray,
    items: List[Dict[str, Any]],
    sum_value0: float,
    w_overlap: float,
    w_value: float,
) -> Dict[str, Any]:
    ov_L, inter_sum, ov_denom, overlap_pairs, _, _ = overlap_loss_0_1(mins, maxs)
    val_L = value_loss_0_1(items, sum_value0)

    ov_term = float(w_overlap) * float(ov_L)
    val_term = float(w_value) * float(val_L)
    core = ov_term + val_term

    return {
        "overlap_L": float(ov_L),
        "value_L": float(val_L),
        "overlap_term": float(ov_term),
        "value_term": float(val_term),
        "core_loss": float(core),
        "inter_sum": float(inter_sum),
        "overlap_denom": float(ov_denom),
        "overlap_pairs": int(overlap_pairs),
    }
