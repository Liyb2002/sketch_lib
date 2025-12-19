#!/usr/bin/env python3
import open3d as o3d
import json
import numpy as np
import os
import random


def get_random_color():
    """Generates a random color (good enough to see touching boundaries)."""
    return [random.random(), random.random(), random.random()]


def run_visualization(json_path, ply_path=None):
    print(f"[Vis] Visualizing primitives from: {os.path.basename(json_path)}")

    if not os.path.exists(json_path):
        print(f"[ERROR] JSON not found: {json_path}")
        return

    if ply_path is not None and not os.path.exists(ply_path):
        print(f"[WARN] PLY not found (will show boxes only): {ply_path}")
        ply_path = None

    with open(json_path, "r") as f:
        data = json.load(f)

    geometries = []

    # 1) Add the point cloud shape first (so boxes overlay)
    if ply_path is not None:
        print(f"  -> Loading shape: {os.path.basename(ply_path)}")
        pcd = o3d.io.read_point_cloud(ply_path)
        geometries.append(pcd)

    # 2) Add Coordinate Frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geometries.append(coord)

    print(f"  -> Drawing {len(data)} boxes (each with a unique color)...")

    for part in data:
        params = part["parameters"]
        center = np.array(params["center"], dtype=np.float64)
        extent = np.array(params["extent"], dtype=np.float64)
        rotation = np.array(params["rotation"], dtype=np.float64)

        # 3) Reconstruct OBB
        obb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)

        # 4) Assign Unique Color
        obb.color = get_random_color()

        geometries.append(obb)

    # 5) Show
    o3d.visualization.draw_geometries(geometries, window_name="PCA Bounding Boxes (Overlay)")


if __name__ == "__main__":
    # Defaults relative to repo layout:
    #   sketch/final_overlays/merged_labeled.ply
    #   sketch/dsl_optimize/pca_primitives.json
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(THIS_DIR)
    SKETCH_ROOT = os.path.join(PROJECT_DIR, "sketch")

    DSL_DIR = os.path.join(SKETCH_ROOT, "dsl_optimize")
    FINAL_DIR = os.path.join(SKETCH_ROOT, "final_overlays")

    json_path = os.path.join(DSL_DIR, "pca_primitives.json")
    ply_path = os.path.join(FINAL_DIR, "merged_labeled.ply")

    run_visualization(json_path, ply_path=ply_path)

