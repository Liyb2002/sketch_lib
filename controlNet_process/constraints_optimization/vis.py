#!/usr/bin/env python3
import json
import os
import open3d as o3d
import numpy as np
from typing import List, Dict, Any, Tuple  # <-- Add the correct import for Tuple

def _load_primitives(primitives_json_path: str) -> List[Dict[str, Any]]:
    with open(primitives_json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "primitives" in data:
        return data["primitives"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected primitives JSON format in {primitives_json_path}")


def _obb_from_params(params: Dict[str, Any]) -> o3d.geometry.OrientedBoundingBox:
    center = np.array(params.get("center", [0, 0, 0]), dtype=np.float64)
    extent = np.array(params.get("extent", [0, 0, 0]), dtype=np.float64)
    R = np.array(params.get("rotation", np.eye(3)), dtype=np.float64)
    return o3d.geometry.OrientedBoundingBox(center, R, extent)


def _load_points_and_cluster_ids(ply_path: str, cluster_ids_path: str) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Missing ply: {ply_path}")
    if not os.path.exists(cluster_ids_path):
        raise FileNotFoundError(f"Missing cluster_ids npy: {cluster_ids_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    cluster_ids = np.load(cluster_ids_path).reshape(-1)
    pts = np.asarray(pcd.points)
    if pts.shape[0] != cluster_ids.shape[0]:
        raise RuntimeError("cluster_ids.npy must align 1:1 with the PLY points.")
    return pcd, cluster_ids


def visualize_initial_bboxes(primitives_json_path: str, ply_path: str, cluster_ids_path: str):
    """
    Visualizes the initial bounding boxes for each label, overlaying each bounding box individually
    with the shape.
    """
    # Load primitives and cluster information
    primitives = _load_primitives(primitives_json_path)

    pcd, cluster_ids = _load_points_and_cluster_ids(ply_path, cluster_ids_path)

    # Iterate over each primitive (bounding box)
    for p in primitives:
        label = p.get("label", "unknown")
        print(f"[VIS] Visualizing label: {label}")

        params = p.get("parameters", {})
        obb = _obb_from_params(params)

        # Colorize bounding boxes and point cloud based on the label
        obb.color = (1.0, 0.6, 0.1)  # Orange color for bounding box
        pcd_colored = _colorize_points_by_cluster_id(pcd, cluster_ids, int(p.get("cluster_id", -1)))

        # Visualize the bounding box and point cloud for this label
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        
        o3d.visualization.draw_geometries(
            [pcd_colored, obb, coord],
            window_name=f"Initial Bounding Box for Label: {label}",
        )


def _colorize_points_by_cluster_id(
    pcd: o3d.geometry.PointCloud,
    cluster_ids: np.ndarray,
    target_cluster_id: int,
    *,
    active_rgb=(1.0, 0.6, 0.1),
    inactive_rgb=(0.05, 0.05, 0.05),
) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    colors = np.zeros((pts.shape[0], 3), dtype=np.float64)
    colors[:] = np.array(inactive_rgb, dtype=np.float64)
    mask = (cluster_ids == target_cluster_id)
    colors[mask] = np.array(active_rgb, dtype=np.float64)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    out.colors = o3d.utility.Vector3dVector(colors)
    return out
