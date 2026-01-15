#!/usr/bin/env python3
import os
import json
import numpy as np
import open3d as o3d

ROOT = os.path.dirname(os.path.abspath(__file__))

SAVE_DIR = os.path.join(ROOT, "sketch", "partfield_overlay", "label_assignment_k20")
PLY_PATH = os.path.join(SAVE_DIR, "assignment_colored.ply")
SEM_JSON = os.path.join(SAVE_DIR, "labels_semantic.json")
IDS_PATH = os.path.join(SAVE_DIR, "assigned_label_ids.npy")

VIS = True


def main():
    # ---- load semantic info ----
    with open(SEM_JSON, "r") as f:
        sem = json.load(f)

    label_id_to_name = {int(k): v for k, v in sem["label_id_to_name"].items()}
    labels_extended = sem["labels_in_order_extended"]

    # ---- load points + assignment ----
    pcd = o3d.io.read_point_cloud(PLY_PATH)
    pts = np.asarray(pcd.points)

    assigned_ids = np.load(IDS_PATH)

    assert pts.shape[0] == assigned_ids.shape[0]

    print(f"Loaded {pts.shape[0]} points, {len(labels_extended)} labels")

    # ---- visualization ----
    if VIS:
        base_gray = np.array([0.25, 0.25, 0.25], dtype=np.float64)

        for lid, name in label_id_to_name.items():
            mask = (assigned_ids == lid)

            if not np.any(mask):
                continue

            colors = np.tile(base_gray[None, :], (pts.shape[0], 1))
            colors[mask] = np.array([1.0, 0.2, 0.2])  # highlight in red

            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(pts)
            pcd_vis.colors = o3d.utility.Vector3dVector(colors)

            title = f"Label: {name}"
            print(f"[VIS] {title}  ({mask.sum()} points)")

            o3d.visualization.draw_geometries([pcd_vis], window_name=title)

    print("Done.")


if __name__ == "__main__":
    main()
