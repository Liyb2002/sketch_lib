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

    DEBUG ADDITIONS (no signature change):
      - Prints why registry cluster ids are not considered / not included.
      - Prints a summary of all skip reasons.
    """
    def _safe_int(x) -> Optional[int]:
        try:
            return int(x)
        except Exception:
            return None

    def _label_for(reg: Dict[str, Any], cid: int) -> str:
        info = reg.get(str(cid), {})
        return str(info.get("label", "unknown"))

    def _print_examples(title: str, cids: List[int], reg: Dict[str, Any], max_n: int = 25):
        if not cids:
            return
        head = cids[:max_n]
        pretty = ", ".join(f"{cid}(label={_label_for(reg, cid)})" for cid in head)
        suffix = "" if len(cids) <= max_n else f" ... (+{len(cids) - max_n} more)"
        print(f"{title}: {pretty}{suffix}")

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

    # ---------------------------
    # DEBUG: registry ids vs npy ids
    # ---------------------------
    registry_ids: List[int] = []
    bad_registry_keys: List[str] = []
    for k in registry.keys():
        cid = _safe_int(k)
        if cid is None:
            bad_registry_keys.append(str(k))
        else:
            registry_ids.append(cid)
    registry_ids = sorted(set(registry_ids))
    reg_set = set(registry_ids)

    unique_clusters = np.unique(cluster_ids)
    unique_clusters = unique_clusters[np.isfinite(unique_clusters)]

    # What will be iterated (post filter)
    if include_negative_cluster_ids:
        unique_iter = unique_clusters
    else:
        unique_iter = unique_clusters[unique_clusters >= 0]

    npy_iter_ids = sorted(set(int(x) for x in unique_iter.tolist()))
    npy_iter_set = set(npy_iter_ids)

    # Registry ids that will never be iterated because missing in npy OR filtered out
    missing_in_npy = sorted(list(reg_set - set(int(x) for x in unique_clusters.tolist())))
    excluded_negative = []
    if not include_negative_cluster_ids:
        excluded_negative = sorted([cid for cid in registry_ids if cid < 0 and cid in set(int(x) for x in unique_clusters.tolist())])

    # Also helpful: ids in registry that are negative (regardless of npy)
    registry_negative_all = sorted([cid for cid in registry_ids if cid < 0])

    print(f"[PCA][DIAG] registry entries: {len(registry)} (int_ids={len(registry_ids)}, bad_keys={len(bad_registry_keys)})")
    if bad_registry_keys:
        print(f"[PCA][DIAG] Non-int registry keys (ignored): {bad_registry_keys[:25]}{' ...' if len(bad_registry_keys) > 25 else ''}")

    print(f"[PCA][DIAG] unique cluster ids in npy (raw, incl negatives): {len(set(int(x) for x in unique_clusters.tolist()))}")
    print(f"[PCA][DIAG] unique cluster ids to iterate (post negative filter): {len(npy_iter_ids)}")

    print(f"[PCA][DIAG] registry ids missing from cluster_ids.npy: {len(missing_in_npy)}")
    _print_examples("[PCA][DIAG] Examples missing_from_npy", missing_in_npy, registry)

    if registry_negative_all:
        print(f"[PCA][DIAG] registry ids that are negative: {len(registry_negative_all)}")
        _print_examples("[PCA][DIAG] Examples registry_negative", registry_negative_all, registry)

    if excluded_negative:
        print(f"[PCA][DIAG] registry ids present in npy but excluded due to negative filter: {len(excluded_negative)}")
        _print_examples("[PCA][DIAG] Examples excluded_negative", excluded_negative, registry)

    # ---------------------------
    # PCA loop
    # ---------------------------
    parts_db: List[Dict[str, Any]] = []

    # Track reasons why an ID that *is in registry* does not produce a primitive.
    per_registry_reason: Dict[int, str] = {}
    for cid in registry_ids:
        if cid in set(int(x) for x in unique_clusters.tolist()):
            if (not include_negative_cluster_ids) and cid < 0:
                per_registry_reason[cid] = "NOT_CONSIDERED: excluded negative cluster id by config"
            else:
                per_registry_reason[cid] = "CONSIDERED: will attempt processing"
        else:
            per_registry_reason[cid] = "NOT_CONSIDERED: cluster id not present in cluster_ids.npy"

    # Counters for skip reasons (for clusters that are iterated)
    skip_counters = {
        "skip_min_points_raw": 0,           # idx.size < min_points
        "skip_all_nonfinite": 0,            # all points non-finite
        "skip_min_points_after_finite": 0,  # finite pts < min_points
        "skip_too_few_unique_points": 0,    # pts_u < 3
    }
    # Keep examples specifically for registry ids
    skipped_registry_examples: Dict[str, List[int]] = {k: [] for k in skip_counters.keys()}

    print(f"[PCA] Found {len(npy_iter_ids)} clusters (including unknown_* if present).")

    for cid in unique_iter:
        cid_int = int(cid)

        idx = np.where(cluster_ids == cid_int)[0]
        if idx.size < min_points:
            skip_counters["skip_min_points_raw"] += 1
            if cid_int in reg_set:
                per_registry_reason[cid_int] = f"SKIPPED: idx.size={int(idx.size)} < min_points={min_points}"
                skipped_registry_examples["skip_min_points_raw"].append(cid_int)
            continue

        pts = points[idx]

        finite_mask = np.isfinite(pts).all(axis=1)
        if not finite_mask.any():
            skip_counters["skip_all_nonfinite"] += 1
            if cid_int in reg_set:
                per_registry_reason[cid_int] = "SKIPPED: all points are non-finite (NaN/Inf)"
                skipped_registry_examples["skip_all_nonfinite"].append(cid_int)
            continue

        pts = pts[finite_mask]
        if pts.shape[0] < min_points:
            skip_counters["skip_min_points_after_finite"] += 1
            if cid_int in reg_set:
                per_registry_reason[cid_int] = f"SKIPPED: finite_points={int(pts.shape[0])} < min_points={min_points}"
                skipped_registry_examples["skip_min_points_after_finite"].append(cid_int)
            continue

        # remove duplicates (critical for qhull)
        pts_u = np.unique(np.round(pts, 6), axis=0)
        if pts_u.shape[0] < 3:
            skip_counters["skip_too_few_unique_points"] += 1
            if cid_int in reg_set:
                per_registry_reason[cid_int] = f"SKIPPED: unique_points={int(pts_u.shape[0])} < 3 after dedup"
                skipped_registry_examples["skip_too_few_unique_points"].append(cid_int)
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

        if cid_int in reg_set:
            per_registry_reason[cid_int] = "INCLUDED: primitive created successfully"

    parts_db.sort(key=lambda d: d["cluster_id"])

    print(f"[PCA] Processed {len(parts_db)} cluster primitives.")

    # ---------------------------
    # DEBUG: print reasons for registry clusters not included
    # ---------------------------
    included_ids = set(d["cluster_id"] for d in parts_db)
    registry_not_included = [cid for cid in registry_ids if cid not in included_ids]

    print("[PCA][DIAG] Registry clusters NOT included in primitives:", len(registry_not_included))
    if registry_not_included:
        # Print detailed reasons (cap at 500 to avoid spam)
        for cid in registry_not_included[:500]:
            print(f"  - {cid} label={_label_for(registry, cid)} => {per_registry_reason.get(cid, 'UNKNOWN_REASON')}")
        if len(registry_not_included) > 500:
            print(f"  ... (+{len(registry_not_included) - 500} more)")

    print(f"[PCA][DIAG] Skip counters (over iterated clusters): {skip_counters}")
    for k, ids in skipped_registry_examples.items():
        _print_examples(f"[PCA][DIAG] Registry examples for {k}", sorted(set(ids)), registry)

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
        include_negative_cluster_ids=False,
    )

    save_primitives_to_json(
        parts_db,
        out_primitives,
        source_ply=os.path.abspath(ply_path),
        source_cluster_ids=os.path.abspath(cluster_ids_path),
    )


if __name__ == "__main__":
    main()
