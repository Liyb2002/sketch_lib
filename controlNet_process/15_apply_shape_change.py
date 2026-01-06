#!/usr/bin/env python3
# 15_apply_shape_change.py
"""
Per-label verification visualization using per-label .ply files:

- Reads sketch/target_edit/all_labels_aabbs.json
- For each label:
    - Loads per-label point cloud from:
        sketch/dsl_optimize/optimize_iteration/iter_000/optimize_results/<label>/*.ply
    - Overlays the label's NEW bounding box:
        - if is_changed: entry["after"]
        - else: entry["aabb"]

Uses original point colors from the .ply.
Bounding box is BLUE for clarity.

Close the window to advance to the next label.
"""

import os
import json
import argparse
from typing import Any, Dict, Tuple, Optional

import numpy as np


# ------------------------ IO ------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def find_first_ply_in_folder(folder: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    plys = [fn for fn in os.listdir(folder) if fn.lower().endswith(".ply")]
    if not plys:
        return None
    plys.sort()
    return os.path.join(folder, plys[0])


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


def get_new_aabb(entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    New bbox definition:
      - changed → after
      - unchanged → aabb
    """
    if entry.get("is_changed", False):
        a = entry["after"]
        return _np3(a["min"]), _np3(a["max"])
    else:
        aabb = entry.get("aabb")
        if aabb is None:
            raise ValueError("Unchanged label missing 'aabb'")
        return _np3(aabb["min"]), _np3(aabb["max"])


def aabb_lineset(mn: np.ndarray, mx: np.ndarray, color_rgb):
    import open3d as o3d

    x0, y0, z0 = mn.tolist()
    x1, y1, z1 = mx.tolist()

    pts = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float64)

    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(edges),
    )

    colors = np.tile(np.array(color_rgb, dtype=np.float64), (edges.shape[0], 1))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


# ------------------------ Open3D helpers ------------------------

def deep_copy_pcd(pcd):
    import open3d as o3d

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())

    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors).copy())
    if pcd.has_normals():
        out.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())

    return out


# ------------------------ Visualization ------------------------

def visualize_label(label: str, ply_path: str, mn: np.ndarray, mx: np.ndarray):
    import open3d as o3d

    pcd = load_ply_pointcloud(ply_path)
    pcd_vis = deep_copy_pcd(pcd)

    # BLUE bounding box
    bbox_color = (0.1, 0.3, 1.0)

    geoms = [
        pcd_vis,
        aabb_lineset(mn, mx, bbox_color),
    ]

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"{label} : per-label PLY + NEW bbox (blue)",
        width=1280,
        height=800,
    )


# ------------------------ Main ------------------------

def main(all_labels_path: str, optimize_results_dir: str):
    data = load_json(all_labels_path)
    labels = data.get("labels", {})
    if not isinstance(labels, dict):
        raise ValueError("Invalid all_labels_aabbs.json: labels must be a dict")

    for label in sorted(labels.keys()):
        entry = labels[label]
        if not isinstance(entry, dict):
            continue

        label_dir = os.path.join(optimize_results_dir, label)
        ply_path = find_first_ply_in_folder(label_dir)
        if ply_path is None:
            print(f"[skip] missing ply for label: {label}")
            continue

        mn, mx = get_new_aabb(entry)
        visualize_label(label, ply_path, mn, mx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all_labels",
        default="sketch/target_edit/all_labels_aabbs.json",
        help="Path to all_labels_aabbs.json",
    )
    parser.add_argument(
        "--optimize_results",
        default="sketch/dsl_optimize/optimize_iteration/iter_000/optimize_results",
        help="Folder with per-label subfolders containing .ply files",
    )
    args = parser.parse_args()

    main(args.all_labels, args.optimize_results)
