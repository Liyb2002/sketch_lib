#!/usr/bin/env python3
import json
import os
from typing import Dict, Any, List, Optional

import numpy as np
import open3d as o3d


def build_registry_from_cluster_map(cluster_map_path: str, out_registry_path: str):
    """
    Converts sketch/clusters/cluster_to_label.json into a registry:
    """
    if not os.path.exists(cluster_map_path):
        raise FileNotFoundError(f"Missing cluster_to_label.json: {cluster_map_path}")

    with open(cluster_map_path, "r") as f:
        cm = json.load(f)

    registry: Dict[str, Dict[str, Any]] = {}

    for k, v in cm.items():
        try:
            cid = int(k)
        except Exception:
            continue

        label = str(v.get("label", "unknown"))
        color_rgb = v.get("color_rgb", None)

        entry = {"label": label}
        if isinstance(color_rgb, (list, tuple)) and len(color_rgb) == 3:
            entry["color_rgb"] = [float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[1])]
        registry[str(cid)] = entry

    os.makedirs(os.path.dirname(out_registry_path), exist_ok=True)
    with open(out_registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"[REGISTRY] Saved: {out_registry_path} ({len(registry)} entries)")


def run_pca_analysis_on_clusters(
    ply_path: str,
    cluster_ids_path: str,
    registry_path: Optional[str] = None,
    min_points: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fits PCA OrientedBoundingBoxes (Open3D) per *cluster id* using cluster_ids.npy.
    Robust to degenerate clusters (line / plane / near-zero extent).
    """
    print(f"[PCA] Loading point cloud: {ply_path}")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Missing ply: {ply_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        print("[PCA] Empty point cloud.")
        return []

    if not os.path.exists(cluster_ids_path):
        raise FileNotFoundError(f"Missing cluster ids npy: {cluster_ids_path}")

    cluster_ids = np.load(cluster_ids_path).reshape(-1)
    if cluster_ids.shape[0] != points.shape[0]:
        raise RuntimeError(
            f"[PCA] cluster_ids length {cluster_ids.shape[0]} != points {points.shape[0]}.\n"
            "cluster_ids.npy must align point-for-point with the PLY."
        )

    registry = {}
    if registry_path is not None:
        if not os.path.exists(registry_path):
            raise FileNotFoundError(f"Registry not found at {registry_path}")
        with open(registry_path, "r") as f:
            registry = json.load(f)

    parts_db: List[Dict[str, Any]] = []

    unique_clusters = np.unique(cluster_ids)
    unique_clusters = unique_clusters[unique_clusters >= 0]

    print(f"[PCA] Found {len(unique_clusters)} clusters (excluding unknown).")

    for cid in unique_clusters:
        idx = np.where(cluster_ids == cid)[0]
        if idx.size < min_points:
            continue

        pts = points[idx]

        # ---- sanitize points ----
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < min_points:
            continue

        # remove duplicates (critical for qhull)
        pts_u = np.unique(np.round(pts, 6), axis=0)
        if pts_u.shape[0] < 3:
            continue

        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(pts_u)

        # ---- detect intrinsic dimension ----
        X = pts_u - pts_u.mean(axis=0, keepdims=True)
        s = np.linalg.svd(X, compute_uv=False)
        rank = int((s > 1e-9).sum())

        # ---- robust OBB ----
        try:
            if rank >= 3:
                obb = pcd_cluster.get_oriented_bounding_box()
            else:
                raise RuntimeError("Degenerate cluster (rank < 3)")
        except Exception:
            aabb = pcd_cluster.get_axis_aligned_bounding_box()
            obb = o3d.geometry.OrientedBoundingBox(
                aabb.get_center(),
                np.eye(3),
                aabb.get_extent(),
            )
            info = registry.get(str(int(cid)), {})
            label = info.get("label", "unknown")
            print(
                f"[PCA][WARN] Fallback AABB for cluster {cid} "
                f"(label={label}, points={pts_u.shape[0]}, rank={rank})"
            )

        info = registry.get(str(int(cid)), {})
        label = info.get("label", "unknown")
        color_rgb = info.get("color_rgb", None)

        record: Dict[str, Any] = {
            "cluster_id": int(cid),
            "label": str(label),
            "parameters": {
                "center": obb.center.tolist(),
                "extent": obb.extent.tolist(),
                "rotation": obb.R.tolist(),
            },
            "point_count": int(idx.size),
        }

        if isinstance(color_rgb, (list, tuple)) and len(color_rgb) == 3:
            record["color_rgb"] = [float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2])]

        parts_db.append(record)

    parts_db.sort(key=lambda d: d["cluster_id"])

    print(f"[PCA] Processed {len(parts_db)} cluster primitives.")
    return parts_db


def save_primitives_to_json(
    parts_db: List[Dict[str, Any]],
    output_path: str,
    source_ply: Optional[str] = None,
):
    """
    Saves primitives to JSON. Optionally stores source_ply for visualization/debug.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if source_ply is not None:
        payload = {
            "source_ply": source_ply,
            "primitives": parts_db,
        }
    else:
        payload = {
            "primitives": parts_db,
        }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[PCA] Saved primitives to: {output_path}")
