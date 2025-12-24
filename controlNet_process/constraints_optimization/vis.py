#!/usr/bin/env python3
"""
constraints_optimization/vis.py

Visualization utilities:
- For each label: load heatmap PLY (colored points) and overlay the PCA OBB box
- Optionally show a combined view of all OBBs together on top of one heatmap (max label)

Reads pca_bboxes.json produced by pca_analysis.py
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _load_colored_ply(path: str) -> "o3d.geometry.PointCloud":
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")
    pcd = o3d.io.read_point_cloud(path)
    return pcd


def _obb_from_dict(obb_dict: Dict[str, Any]) -> "o3d.geometry.OrientedBoundingBox":
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")
    center = np.asarray(obb_dict["center"], dtype=np.float64)
    R = np.asarray(obb_dict["R"], dtype=np.float64)
    extent = np.asarray(obb_dict["extent"], dtype=np.float64)
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    return obb


def _lineset_from_obb(obb: "o3d.geometry.OrientedBoundingBox") -> "o3d.geometry.LineSet":
    """
    Create a LineSet for drawing the OBB.
    """
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    # Use a bright color (yellow-ish) without hardcoding exact style; Open3D needs explicit color
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 1.0, 0.0]]), (len(ls.lines), 1)))
    return ls


def visualize_heatmaps_with_bboxes(
    *,
    heat_dir: str,
    bbox_json: str,
    max_labels_to_show: int = 12,
    show_combined: bool = True,
) -> None:
    """
    Pops Open3D windows. For each label:
      - show heatmap PLY (colored)
      - overlay bbox LineSet

    If show_combined:
      show a single window with multiple bboxes overlaid on top of the largest label heatmap.
    """
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")

    if not os.path.exists(bbox_json):
        raise FileNotFoundError(f"Missing bbox_json: {bbox_json}")

    payload = _load_json(bbox_json)
    labels: List[Dict[str, Any]] = list(payload.get("labels", []))
    if not labels:
        print("[VIS] No labels to visualize (bbox_json has empty labels).")
        return

    # Sort by points_used descending
    labels = sorted(labels, key=lambda r: int(r.get("points_used", 0)), reverse=True)

    to_show = labels[: int(max_labels_to_show)]

    # Per-label windows
    for rec in to_show:
        label = str(rec.get("label", rec.get("sanitized", "unknown")))
        heat_ply = str(rec["heat_ply"])
        obb_dict = rec["obb"]

        pcd = _load_colored_ply(heat_ply)
        obb = _obb_from_dict(obb_dict)
        ls = _lineset_from_obb(obb)

        title = f"HeatMap + PCA-BBox | {label} | points_used={rec.get('points_used')} | min_heat={rec.get('min_heat')}"
        print("[VIS] opening:", title)
        o3d.visualization.draw_geometries([pcd, ls], window_name=title)

    if not show_combined:
        return

    # Combined view: overlay all shown bboxes on top of the largest label heatmap
    base = to_show[0]
    base_ply = str(base["heat_ply"])
    base_pcd = _load_colored_ply(base_ply)

    geoms = [base_pcd]
    for rec in to_show:
        obb = _obb_from_dict(rec["obb"])
        geoms.append(_lineset_from_obb(obb))

    title = f"COMBINED | {len(to_show)} bboxes over base heatmap ({base.get('label')})"
    print("[VIS] opening:", title)
    o3d.visualization.draw_geometries(geoms, window_name=title)
