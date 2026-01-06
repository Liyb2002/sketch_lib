#!/usr/bin/env python3
# 15_apply_shape_change.py
"""
Apply per-label bbox change to per-label point clouds (verification):

For each label folder:
  - Load per-label .ply (points + ORIGINAL colors)
  - Read original/new AABBs from all_labels_aabbs.json
  - Compute a per-axis scaling + translation that maps original bbox -> new bbox:
        p_scaled = (p - c_old) * s + c_old          (scaling first, about old center)
        p_new    = p_scaled + (c_new - c_old)       (then translation)
    where s = (extent_new / extent_old) elementwise

  - Visualize:
      - Original point cloud (dimmed colors)
      - Transformed point cloud (original colors)
      - Original bbox (blue)
      - New bbox (red)

Close the window to advance to the next label.

No saving.
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


def get_original_and_new_aabb(entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      (orig_min, orig_max, new_min, new_max)

    - If changed: original=before, new=after
    - If unchanged: original=aabb, new=aabb
    """
    if entry.get("is_changed", False):
        b = entry.get("before", None)
        a = entry.get("after", None)
        if not (isinstance(b, dict) and isinstance(a, dict)):
            raise ValueError("Entry marked changed but missing before/after.")
        return _np3(b["min"]), _np3(b["max"]), _np3(a["min"]), _np3(a["max"])
    else:
        aabb = entry.get("aabb", None)
        if not isinstance(aabb, dict):
            raise ValueError("Unchanged entry missing 'aabb'.")
        mn = _np3(aabb["min"])
        mx = _np3(aabb["max"])
        return mn, mx, mn.copy(), mx.copy()


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

def pcd_from_points_colors(points: np.ndarray, colors: Optional[np.ndarray]):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def get_points_colors(pcd):
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
    return pts, cols


# ------------------------ Transform (scale then translate) ------------------------

def compute_scale_and_translation(
    orig_mn: np.ndarray, orig_mx: np.ndarray,
    new_mn: np.ndarray, new_mx: np.ndarray,
    eps: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      c_old, c_new, s, t
    where
      s = extent_new / extent_old (per-axis), with safe handling for tiny extents
      t = c_new - c_old
    """
    c_old = 0.5 * (orig_mn + orig_mx)
    c_new = 0.5 * (new_mn + new_mx)

    ext_old = (orig_mx - orig_mn)
    ext_new = (new_mx - new_mn)

    s = np.ones(3, dtype=np.float64)
    for i in range(3):
        if abs(ext_old[i]) > eps:
            s[i] = ext_new[i] / ext_old[i]
        else:
            s[i] = 1.0  # degenerate axis: don't scale

    t = (c_new - c_old)
    return c_old, c_new, s, t


def apply_scale_then_translate(points: np.ndarray, c_old: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    scaling first about c_old, then translation by t.
    """
    return (points - c_old[None, :]) * s[None, :] + c_old[None, :] + t[None, :]


# ------------------------ Visualization ------------------------

def visualize_label_applied(
    label: str,
    ply_path: str,
    orig_mn: np.ndarray, orig_mx: np.ndarray,
    new_mn: np.ndarray, new_mx: np.ndarray,
):
    import open3d as o3d

    src = load_ply_pointcloud(ply_path)
    pts, cols = get_points_colors(src)

    # compute transform from bbox change
    c_old, c_new, s, t = compute_scale_and_translation(orig_mn, orig_mx, new_mn, new_mx)
    pts_new = apply_scale_then_translate(pts, c_old, s, t)

    # build pcds
    pcd_new = pcd_from_points_colors(pts_new, cols)

    # show original too (dim colors so it doesn't dominate)
    if cols is not None:
        cols_dim = np.clip(cols * 0.25, 0.0, 1.0)
    else:
        cols_dim = None
    pcd_old = pcd_from_points_colors(pts, cols_dim)

    blue = (0.1, 0.3, 1.0)  # original bbox
    red  = (1.0, 0.2, 0.2)  # new bbox

    geoms = [
        pcd_old,
        pcd_new,
        aabb_lineset(orig_mn, orig_mx, blue),
        aabb_lineset(new_mn,  new_mx,  red),
    ]

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"{label} : applied (scale->translate)  | bbox blue=orig red=new",
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

        orig_mn, orig_mx, new_mn, new_mx = get_original_and_new_aabb(entry)
        visualize_label_applied(label, ply_path, orig_mn, orig_mx, new_mn, new_mx)


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
