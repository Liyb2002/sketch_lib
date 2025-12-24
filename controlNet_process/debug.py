#!/usr/bin/env python3
import os
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PLY_PATH = os.path.join(ROOT_DIR, "sketch", "3d_reconstruction", "fused_model.ply")

BOX_FRAC = 0.015   # << what you asked for


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def make_wire_bbox(min_pt, max_pt, color):
    """
    Create a wireframe AABB as a LineSet.
    """
    min_pt = np.asarray(min_pt)
    max_pt = np.asarray(max_pt)

    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
    ])

    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]

    colors = [color for _ in lines]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    if not os.path.exists(PLY_PATH):
        raise RuntimeError(f"Missing ply: {PLY_PATH}")

    print(f"[INFO] Loading shape: {PLY_PATH}")
    pcd = o3d.io.read_point_cloud(PLY_PATH)
    pts = np.asarray(pcd.points)

    shape_min = pts.min(axis=0)
    shape_max = pts.max(axis=0)
    dims = shape_max - shape_min
    avg_dim = (dims[0] + dims[1] + dims[2]) / 3.0

    box_size = BOX_FRAC * avg_dim
    half = box_size * 0.5

    center = (shape_min + shape_max) * 0.5
    small_min = center - half
    small_max = center + half

    print("[INFO] Shape bbox dims:", dims)
    print("[INFO] Avg dim:", avg_dim)
    print("[INFO] Small box frac:", BOX_FRAC)
    print("[INFO] Small box side length:", box_size)

    # --- geometry ---
    shape_bbox = make_wire_bbox(shape_min, shape_max, color=[0, 0, 0])   # black
    small_box  = make_wire_bbox(small_min, small_max, color=[1, 0, 0])   # red

    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    o3d.visualization.draw_geometries(
        [pcd, shape_bbox, small_box],
        window_name="Shape + global bbox + small cube",
        width=1200,
        height=900
    )


if __name__ == "__main__":
    main()
