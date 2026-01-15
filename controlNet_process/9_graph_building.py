#!/usr/bin/env python3
import os
import json
import numpy as np
import open3d as o3d

from graph_building.pca_analysis import run as run_pca

ROOT = os.path.dirname(os.path.abspath(__file__))

SAVE_DIR = os.path.join(ROOT, "sketch", "partfield_overlay", "label_assignment_k20")
PLY_PATH = os.path.join(SAVE_DIR, "assignment_colored.ply")
SEM_JSON = os.path.join(SAVE_DIR, "labels_semantic.json")
IDS_PATH = os.path.join(SAVE_DIR, "assigned_label_ids.npy")

VIS = True


def make_obb_lines(center, axes, extents, color=(0, 1, 0)):
    corners = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                corners.append(
                    center + dx * axes[:, 0] * extents[0]
                           + dy * axes[:, 1] * extents[1]
                           + dz * axes[:, 2] * extents[2]
                )
    corners = np.array(corners)

    edges = [
        (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),
        (3,7),(4,5),(4,6),(5,7),(6,7)
    ]

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(corners)
    lines.lines = o3d.utility.Vector2iVector(edges)
    lines.colors = o3d.utility.Vector3dVector([color]*len(edges))
    return lines


def main():
    # --- compute PCA bboxes first ---
    bbox_json_path = run_pca(SAVE_DIR)

    with open(bbox_json_path, "r") as f:
        bboxes = json.load(f)

    with open(SEM_JSON, "r") as f:
        sem = json.load(f)

    label_id_to_name = {int(k): v for k, v in sem["label_id_to_name"].items()}

    pcd = o3d.io.read_point_cloud(PLY_PATH)
    pts = np.asarray(pcd.points)
    assigned_ids = np.load(IDS_PATH)

    if not VIS:
        return

    base_gray = np.array([0.25, 0.25, 0.25])

    for name, info in bboxes.items():
        lid = info["label_id"]
        mask = (assigned_ids == lid)

        colors = np.tile(base_gray[None, :], (pts.shape[0], 1))
        colors[mask] = np.array([1.0, 0.2, 0.2])

        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(pts)
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)

        obb = info["obb_pca"]
        lines = make_obb_lines(
            np.array(obb["center"]),
            np.array(obb["axes"]),
            np.array(obb["extents"])
        )

        title = f"{name}  (points={info['n_points']})"
        print("[VIS]", title)

        o3d.visualization.draw_geometries([pcd_vis, lines], window_name=title)


if __name__ == "__main__":
    main()
