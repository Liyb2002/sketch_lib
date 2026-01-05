#!/usr/bin/env python3
# 15_apply_shape_change.py
"""
Verification-only visualization:
- Loads the original fused point cloud: sketch/3d_reconstruction/fused_model.ply
- Reads sketch/target_edit/all_labels_aabbs.json
- Finds labels with is_changed == True
- Overlays their BEFORE AABBs and AFTER AABBs on the SAME original point cloud.

No saving. No printing.
"""

import os
import json
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np


# ------------------------ IO ------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_ply_pointcloud(path: str):
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(
            "open3d is required for visualization.\n"
            "Install: pip install open3d\n"
            f"Original import error: {e}"
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(f"PLY not found: {path}")

    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Loaded PLY but point cloud is empty: {path}")
    return pcd


# ------------------------ AABB utils ------------------------

def _np3(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)


def aabb_lineset(mn: np.ndarray, mx: np.ndarray, color_rgb: Tuple[float, float, float]):
    """
    Create a wireframe AABB LineSet for Open3D.
    """
    import open3d as o3d

    mn = mn.astype(np.float64)
    mx = mx.astype(np.float64)

    # 8 corners
    x0, y0, z0 = mn.tolist()
    x1, y1, z1 = mx.tolist()
    pts = np.array([
        [x0, y0, z0],  # 0
        [x1, y0, z0],  # 1
        [x1, y1, z0],  # 2
        [x0, y1, z0],  # 3
        [x0, y0, z1],  # 4
        [x1, y0, z1],  # 5
        [x1, y1, z1],  # 6
        [x0, y1, z1],  # 7
    ], dtype=np.float64)

    # 12 edges
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
    ], dtype=np.int32)

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(edges),
    )
    colors = np.tile(np.array(color_rgb, dtype=np.float64)[None, :], (edges.shape[0], 1))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def extract_changed_aabbs(labels_dict: Dict[str, Any]) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns list of tuples:
      (label, before_min, before_max, after_min, after_max)
    """
    out = []
    for label, entry in labels_dict.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("is_changed", False) is True:
            b = entry.get("before", None)
            a = entry.get("after", None)
            if not (isinstance(b, dict) and isinstance(a, dict)):
                continue
            out.append((
                label,
                _np3(b["min"]), _np3(b["max"]),
                _np3(a["min"]), _np3(a["max"]),
            ))
    return out


# ------------------------ Visualization ------------------------

def deep_copy_pcd(pcd):
    """
    Open3D version-agnostic deep copy (works even when pcd.clone() doesn't exist).
    """
    import open3d as o3d

    pcd_copy = o3d.geometry.PointCloud()
    pcd_copy.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())

    if pcd.has_colors():
        pcd_copy.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors).copy())
    if pcd.has_normals():
        pcd_copy.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())

    return pcd_copy


def visualize_overlay(pcd, changed: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    import open3d as o3d

    # Deep copy so we don't mutate the original pcd object
    pcd_vis = deep_copy_pcd(pcd)

    # Make the point cloud a neutral gray for readability
    gray = np.array([0.65, 0.65, 0.65], dtype=np.float64)
    pts = np.asarray(pcd_vis.points)
    pcd_vis.colors = o3d.utility.Vector3dVector(np.tile(gray[None, :], (pts.shape[0], 1)))

    geoms = [pcd_vis]

    # BEFORE boxes (blue-ish), AFTER boxes (red-ish)
    before_color = (0.2, 0.4, 1.0)
    after_color  = (1.0, 0.2, 0.2)

    for (_label, bmn, bmx, amn, amx) in changed:
        geoms.append(aabb_lineset(bmn, bmx, before_color))
        geoms.append(aabb_lineset(amn, amx, after_color))

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Verification Overlay: BEFORE (blue) vs AFTER (red) AABBs on original fused_model",
        width=1280,
        height=800,
    )


def main(ply_path: str, all_labels_path: str):
    pcd = load_ply_pointcloud(ply_path)
    data = load_json(all_labels_path)

    labels_dict = data.get("labels", {})
    if not isinstance(labels_dict, dict):
        raise ValueError(f"Invalid all_labels_aabbs.json: 'labels' must be a dict: {all_labels_path}")

    changed = extract_changed_aabbs(labels_dict)

    # If nothing changed, show nothing is confusing — fail loudly
    if len(changed) == 0:
        raise RuntimeError("No changed labels found (labels[*].is_changed == True). Nothing to visualize.")

    visualize_overlay(pcd, changed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ply",
        default="sketch/3d_reconstruction/fused_model.ply",
        help="Path to the original fused model PLY",
    )
    parser.add_argument(
        "--all_labels",
        default="sketch/target_edit/all_labels_aabbs.json",
        help="Path to all_labels_aabbs.json",
    )
    args = parser.parse_args()

    main(args.ply, args.all_labels)
