#!/usr/bin/env python3
"""
constraints_optimization/save_new_segmentation.py

Minimal viewer:
- Finds each label folder under: heat_dir/heatmaps/<label>/
- Loads the first file matching: heat_map_*.ply
- Visualizes it with Open3D (one window at a time).
"""

import os

try:
    import open3d as o3d
except Exception:
    o3d = None


def _find_heatmap_ply_per_label(heat_dir: str):
    """
    Returns list of tuples: (label_folder_name, ply_path)
    where ply_path is the first sorted heat_map_*.ply in that folder.
    """
    heatmaps_root = os.path.join(heat_dir, "heatmaps")
    if not os.path.isdir(heatmaps_root):
        raise FileNotFoundError(f"Missing heatmaps folder: {heatmaps_root}")

    out = []
    for sub in sorted(os.listdir(heatmaps_root)):
        subdir = os.path.join(heatmaps_root, sub)
        if not os.path.isdir(subdir):
            continue

        candidates = [f for f in os.listdir(subdir) if f.startswith("heat_map_") and f.endswith(".ply")]
        if not candidates:
            continue
        candidates.sort()

        out.append((sub, os.path.join(subdir, candidates[0])))

    if not out:
        raise FileNotFoundError(f"No heat_map_*.ply found under: {heatmaps_root}")
    return out


def vis_heatmap_plys_per_label(*, heat_dir: str) -> None:
    """
    Visualize each heatmap PLY one-by-one.
    Close the Open3D window to proceed to the next label.
    """
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")

    entries = _find_heatmap_ply_per_label(heat_dir)

    print(f"[HEAT_VIS] found {len(entries)} labels under: {os.path.join(heat_dir, 'heatmaps')}")
    for i, (label, ply_path) in enumerate(entries):
        print(f"\n[HEAT_VIS] ({i+1}/{len(entries)}) label: {label}")
        print(f"[HEAT_VIS] ply  : {os.path.abspath(ply_path)}")

        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty():
            print("[HEAT_VIS] WARNING: empty point cloud, skipping.")
            continue

        # Uses the colors stored in the PLY already
        o3d.visualization.draw_geometries([pcd])
