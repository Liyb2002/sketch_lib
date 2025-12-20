#!/usr/bin/env python3
import json
import os
import random
from typing import Optional, List, Dict, Any

import numpy as np
import open3d as o3d


def get_random_color():
    return [random.random(), random.random(), random.random()]


def _load_primitives(primitives_json_path: str) -> List[Dict[str, Any]]:
    with open(primitives_json_path, "r") as f:
        data = json.load(f)

    # New format: {"source_ply": "...", "primitives": [...]}
    if isinstance(data, dict) and "primitives" in data:
        return data["primitives"]

    # Old format: [...]
    if isinstance(data, list):
        return data

    raise ValueError(f"Unexpected primitives JSON format in {primitives_json_path}")


def run_visualization(primitives_json_path: str, ply_path: Optional[str] = None):
    print(f"[Vis] Visualizing primitives from: {os.path.basename(primitives_json_path)}")

    if not os.path.exists(primitives_json_path):
        print(f"[ERROR] JSON not found: {primitives_json_path}")
        return

    parts = _load_primitives(primitives_json_path)

    geometries = []

    # Load shape
    if ply_path is not None and os.path.exists(ply_path):
        print(f"  -> Loading shape: {os.path.basename(ply_path)}")
        pcd = o3d.io.read_point_cloud(ply_path)
        geometries.append(pcd)
    else:
        if ply_path is None:
            print("[WARN] No ply_path provided; showing boxes only.")
        else:
            print(f"[WARN] PLY not found: {ply_path}; showing boxes only.")

    # Coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geometries.append(coord)

    print(f"  -> Drawing {len(parts)} PCA bounding boxes...")

    for part in parts:
        params = part.get("parameters", {})
        center = np.array(params.get("center", [0, 0, 0]), dtype=np.float64)
        extent = np.array(params.get("extent", [0, 0, 0]), dtype=np.float64)
        rotation = np.array(params.get("rotation", np.eye(3)), dtype=np.float64)

        obb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
        obb.color = get_random_color()
        geometries.append(obb)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="PCA Bounding Boxes over Shape (per-cluster)",
    )


if __name__ == "__main__":
    # Default behavior if run standalone:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(THIS_DIR)
    SKETCH_ROOT = os.path.join(PROJECT_DIR, "sketch")

    primitives_json = os.path.join(SKETCH_ROOT, "dsl_optimize", "pca_primitives.json")
    default_ply = os.path.join(SKETCH_ROOT, "clusters", "labeled_clusters.ply")

    run_visualization(primitives_json, ply_path=default_ply)
