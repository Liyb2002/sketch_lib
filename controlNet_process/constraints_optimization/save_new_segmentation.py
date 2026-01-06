#!/usr/bin/env python3
"""
constraints_optimization/save_new_segmentation.py

Visualization helpers for:
- per-label heatmap PLYs (stored in entry["heat_ply"])
- optimized world AABBs (stored in entry["opt_aabb_world"])

Input JSON format: pca_bboxes_optimized.json (same schema as your example).
"""

import os
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# -------------------- IO --------------------

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


# -------------------- Open3D helpers --------------------

def _aabb_from_minmax(mn: List[float], mx: List[float]) -> "o3d.geometry.AxisAlignedBoundingBox":
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")
    mnv = np.array(mn, dtype=np.float64)
    mxv = np.array(mx, dtype=np.float64)
    return o3d.geometry.AxisAlignedBoundingBox(min_bound=mnv, max_bound=mxv)


def _set_color(geom: Any, rgb: Tuple[float, float, float]) -> None:
    # Works for AxisAlignedBoundingBox / OrientedBoundingBox
    geom.color = np.array(rgb, dtype=np.float64)


def _obb_from_center_R_extent(center, R, extent) -> "o3d.geometry.OrientedBoundingBox":
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")
    c = np.array(center, dtype=np.float64)
    RR = np.array(R, dtype=np.float64)
    e = np.array(extent, dtype=np.float64)
    return o3d.geometry.OrientedBoundingBox(center=c, R=RR, extent=e)


# -------------------- public API --------------------

def vis_all_labels_heatmap_with_opt_aabb(
    *,
    pca_bboxes_optimized_json: str,
    prefer_opt_aabb: bool = True,
    show_obb: bool = False,
) -> None:
    """
    For each entry in payload["labels"]:
      - load entry["heat_ply"]
      - draw heatmap point cloud + AABB (opt_aabb_world or aabb)

    AABB choice:
      - if prefer_opt_aabb and opt_aabb_world exists -> use it
      - else -> fallback to entry["aabb"] (min_bound/max_bound)

    If heatmap PLY missing, prints:
      [MISS] <label> : heatmap ply not found at <path>
    """
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")

    payload = _load_json(pca_bboxes_optimized_json)
    entries = payload.get("labels", []) or []
    if not isinstance(entries, list):
        raise ValueError("Expected payload['labels'] to be a list.")

    print(f"[VIS] entries: {len(entries)}")
    for i, rec in enumerate(entries):
        if not isinstance(rec, dict):
            continue

        label = str(rec.get("label", "unknown"))
        heat_ply = rec.get("heat_ply", None)

        print(f"\n[VIS] ({i+1}/{len(entries)}) label: {label}")

        if not isinstance(heat_ply, str) or not heat_ply:
            print(f"[MISS] {label} : missing 'heat_ply' field")
            continue

        if not os.path.exists(heat_ply):
            print(f"[MISS] {label} : heatmap ply not found at {heat_ply}")
            continue

        pcd = o3d.io.read_point_cloud(heat_ply)
        if pcd.is_empty():
            print(f"[VIS] WARNING: empty point cloud, skipping draw.")
            continue

        geoms: List[Any] = [pcd]

        # Choose which AABB to show
        aabb_added = False
        if prefer_opt_aabb and isinstance(rec.get("opt_aabb_world", None), dict):
            ob = rec["opt_aabb_world"]
            if "min" in ob and "max" in ob:
                aabb = _aabb_from_minmax(ob["min"], ob["max"])
                _set_color(aabb, (0.0, 1.0, 0.0))  # green
                geoms.append(aabb)
                aabb_added = True

        if not aabb_added:
            ab = rec.get("aabb", None)
            if isinstance(ab, dict) and "min_bound" in ab and "max_bound" in ab:
                aabb = _aabb_from_minmax(ab["min_bound"], ab["max_bound"])
                _set_color(aabb, (0.0, 0.0, 1.0))  # blue
                geoms.append(aabb)
                aabb_added = True
            else:
                print(f"[VIS] WARNING: no usable AABB found for {label} (showing heatmap only)")

        # Optional: show OBB too
        if show_obb:
            obb = rec.get("obb", None)
            if isinstance(obb, dict) and all(k in obb for k in ("center", "R", "extent")):
                try:
                    obb_geom = _obb_from_center_R_extent(obb["center"], obb["R"], obb["extent"])
                    _set_color(obb_geom, (1.0, 0.0, 0.0))  # red
                    geoms.append(obb_geom)
                except Exception as e:
                    print(f"[VIS] WARNING: failed to build OBB for {label}: {e}")

        print("[VIS] heat_ply:", heat_ply)
        o3d.visualization.draw_geometries(geoms)
