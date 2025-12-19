#!/usr/bin/env python3
import open3d as o3d
import json
import numpy as np
import os
import random


def get_random_color():
    """Generates a random color (good for distinguishing touching boxes)."""
    return [random.random(), random.random(), random.random()]


def run_visualization(primitives_json_path):
    print(f"[Vis] Visualizing primitives from: {os.path.basename(primitives_json_path)}")

    if not os.path.exists(primitives_json_path):
        print(f"[ERROR] JSON not found: {primitives_json_path}")
        return

    # ------------------------------------------------------------
    # Resolve paths internally (no dependency on main.py)
    # ------------------------------------------------------------
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(THIS_DIR)
    SKETCH_ROOT = os.path.join(PROJECT_DIR, "sketch")

    MERGED_PLY = os.path.join(SKETCH_ROOT, "final_overlays", "merged_labeled.ply")

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    with open(primitives_json_path, "r") as f:
        parts = json.load(f)

    geometries = []

    # 1) Load and add the point cloud shape
    if os.path.exists(MERGED_PLY):
        print(f"  -> Loading shape: {os.path.basename(MERGED_PLY)}")
        pcd = o3d.io.read_point_cloud(MERGED_PLY)
        geometries.append(pcd)
    else:
        print(f"[WARN] merged_labeled.ply not found, showing boxes only")

    # 2) Add coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    geometries.append(coord)

    print(f"  -> Drawing {len(parts)} PCA bounding boxes...")

    # 3) Add PCA bounding boxes
    for part in parts:
        params = part["parameters"]

        center = np.array(params["center"], dtype=np.float64)
        extent = np.array(params["extent"], dtype=np.float64)
        rotation = np.array(params["rotation"], dtype=np.float64)

        obb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
        obb.color = get_random_color()

        geometries.append(obb)

    # 4) Show
    o3d.visualization.draw_geometries(
        geometries,
        window_name="PCA Bounding Boxes over Shape",
    )


if __name__ == "__main__":
    # Default location
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(THIS_DIR)
    SKETCH_ROOT = os.path.join(PROJECT_DIR, "sketch")

    primitives_json = os.path.join(
        SKETCH_ROOT, "dsl_optimize", "pca_primitives.json"
    )

    run_visualization(primitives_json)
