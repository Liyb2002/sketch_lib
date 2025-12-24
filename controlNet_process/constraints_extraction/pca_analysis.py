#!/usr/bin/env python3
import json
import os
from typing import Dict, Any, List, Optional

import numpy as np
import open3d as o3d


def build_registry_from_cluster_map(cluster_map_path: str, out_registry_path: str):
    """
    Converts sketch/clusters/cluster_to_label.json into a registry.json:
      registry[cluster_id_str] = {"label": "...", "color_rgb": [r,g,b]}
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
            # FIX: use (0,1,2) not (0,1,1)
            entry["color_rgb"] = [float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2])]

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
    include_negative_cluster_ids: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fits PCA OrientedBoundingBoxes (Open3D) per *cluster id* using cluster_ids.npy.

    IMPORTANT: runs on ALL clusters INCLUDING unknown_* (which are normal cluster ids >= 0).
    Only excludes negative cluster ids unless include_negative_cluster_ids=True.
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

    registry: Dict[str, Any] = {}
    if registry_path is not None:
        if not os.path.exists(registry_path):
            raise FileNotFoundError(f"Registry not found at {registry_path}")
        with open(registry_path, "r") as f:
            registry = json.load(f)

    parts_db: List[Dict[str, Any]] = []

    unique_clusters = np.unique(cluster_ids)
    if include_negative_cluster_ids:
        unique_clusters = unique_clusters[np.isfinite(unique_clusters)]
    else:
        unique_clusters = unique_clusters[unique_clusters >= 0]

    # NOTE: We do NOT exclude "unknown" here. Unknowns are just labels in registry.
    print(f"[PCA] Found {len(unique_clusters)} clusters (including unknown_* if present).")

    for cid in unique_clusters:
        cid_int = int(cid)

        idx = np.where(cluster_ids == cid_int)[0]
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
        used_fallback_aabb = False
        try:
            if rank >= 3:
                obb = pcd_cluster.get_oriented_bounding_box()
            else:
                raise RuntimeError("Degenerate cluster (rank < 3)")
        except Exception:
            used_fallback_aabb = True
            aabb = pcd_cluster.get_axis_aligned_bounding_box()
            obb = o3d.geometry.OrientedBoundingBox(
                aabb.get_center(),
                np.eye(3),
                aabb.get_extent(),
            )

        info = registry.get(str(cid_int), {})
        label = str(info.get("label", "unknown"))
        color_rgb = info.get("color_rgb", None)

        if used_fallback_aabb:
            print(
                f"[PCA][WARN] Fallback AABB for cluster {cid_int} "
                f"(label={label}, points={pts_u.shape[0]}, rank={rank})"
            )

        record: Dict[str, Any] = {
            "cluster_id": cid_int,
            "label": label,
            "parameters": {
                "center": obb.center.tolist(),
                "extent": obb.extent.tolist(),
                "rotation": obb.R.tolist(),
            },
            "point_count": int(idx.size),
            "rank": int(rank),
            "used_fallback_aabb": bool(used_fallback_aabb),
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
    source_cluster_ids: Optional[str] = None,
):
    """
    Saves primitives to JSON. Optionally stores source paths for visualization/debug.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload: Dict[str, Any] = {"primitives": parts_db}
    if source_ply is not None:
        payload["source_ply"] = source_ply
    if source_cluster_ids is not None:
        payload["source_cluster_ids"] = source_cluster_ids

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[PCA] Saved primitives to: {output_path}")


# Optional CLI helper
def main():
    """
    Example expected layout:
      sketch/clusters/labeled_clusters.ply
      sketch/clusters/final_cluster_ids.npy
      sketch/clusters/cluster_to_label.json
      sketch/clusters/registry.json
      sketch/clusters/pca_primitives.json
    """
    root = os.path.dirname(os.path.abspath(__file__))
    clusters_dir = os.path.join(root, "sketch", "clusters")

    ply_path = os.path.join(clusters_dir, "labeled_clusters.ply")
    cluster_ids_path = os.path.join(clusters_dir, "final_cluster_ids.npy")
    cluster_map_path = os.path.join(clusters_dir, "cluster_to_label.json")
    registry_path = os.path.join(clusters_dir, "registry.json")
    out_primitives = os.path.join(clusters_dir, "pca_primitives.json")

    # Build registry from cluster_to_label.json
    build_registry_from_cluster_map(cluster_map_path, registry_path)

    # Run PCA on ALL clusters (including unknown_*)
    parts_db = run_pca_analysis_on_clusters(
        ply_path=ply_path,
        cluster_ids_path=cluster_ids_path,
        registry_path=registry_path,
        min_points=10,
        include_negative_cluster_ids=False,  # should already be none if your mapping fixed -1
    )

    save_primitives_to_json(
        parts_db,
        out_primitives,
        source_ply=os.path.abspath(ply_path),
        source_cluster_ids=os.path.abspath(cluster_ids_path),
    )


if __name__ == "__main__":
    main()
