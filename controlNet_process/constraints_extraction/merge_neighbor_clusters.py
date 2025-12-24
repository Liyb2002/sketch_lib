#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import open3d as o3d
from collections import defaultdict, deque

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _load_registry(registry_path: str):
    with open(registry_path, "r") as f:
        reg = json.load(f)

    reg_int = {}
    for k, v in reg.items():
        try:
            cid = int(k)
        except Exception:
            continue
        reg_int[cid] = v
    return reg_int


def _cluster_points(points: np.ndarray, cluster_ids: np.ndarray):
    clusters = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        cid = int(cid)
        if cid < 0:
            continue
        clusters[cid].append(i)
    return {cid: np.array(idxs, dtype=np.int64) for cid, idxs in clusters.items()}


def _aabb_from_points(pts: np.ndarray):
    mn = np.min(pts, axis=0)
    mx = np.max(pts, axis=0)
    return mn, mx


def _aabb_min_distance(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    dx = max(0.0, float(b_min[0] - a_max[0]), float(a_min[0] - b_max[0]))
    dy = max(0.0, float(b_min[1] - a_max[1]), float(a_min[1] - b_max[1]))
    dz = max(0.0, float(b_min[2] - a_max[2]), float(a_min[2] - b_max[2]))
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def _is_unknown_label(lab: str) -> bool:
    s = str(lab).strip().lower()
    return s == "unknown" or s.startswith("unknown_")


def _base_label(label: str) -> str:
    s = str(label)
    if "_" not in s:
        return s
    a, b = s.rsplit("_", 1)
    try:
        int(b)
        return a
    except Exception:
        return s


def _is_degenerate_cluster_points(pts: np.ndarray, eps: float = 1e-12) -> bool:
    """
    Degenerate if:
      - no points
      - all points identical (after rounding)
      - AABB extent is ~0 in all axes
    """
    if pts is None:
        return True
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return True

    # All identical? (robust via rounding)
    uniq = np.unique(np.round(pts, 8), axis=0)
    if uniq.shape[0] <= 1:
        return True

    mn, mx = _aabb_from_points(pts)
    ext = mx - mn
    if not np.isfinite(ext).all():
        return True
    if float(np.max(ext)) <= eps:
        return True

    return False


def _safe_obb_or_aabb_params(pts: np.ndarray):
    """
    Try OBB. If it fails/degenerate, fall back to AABB with identity rotation.
    """
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]

    if pts.shape[0] == 0:
        return None  # caller decides what to do

    # Try OBB
    try:
        temp = o3d.geometry.PointCloud()
        temp.points = o3d.utility.Vector3dVector(pts)
        obb = temp.get_oriented_bounding_box()

        ext = np.array(obb.extent, dtype=np.float64)
        if np.isfinite(ext).all() and np.min(ext) >= 1e-12:
            return {
                "center": obb.center.tolist(),
                "extent": obb.extent.tolist(),
                "rotation": obb.R.tolist(),
                "bbox_mode": "obb",
            }
    except Exception:
        pass

    # Fallback AABB
    mn, mx = _aabb_from_points(pts)
    center = ((mn + mx) * 0.5).tolist()
    extent = (mx - mn).tolist()
    if not np.isfinite(extent).all() or float(np.max(mx - mn)) <= 1e-12:
        return None

    return {
        "center": center,
        "extent": extent,
        "rotation": np.eye(3).tolist(),
        "bbox_mode": "aabb_fallback",
    }


# -----------------------------------------------------------------------------
# Graph building (known labels only)
# -----------------------------------------------------------------------------
def build_neighbor_graph(
    ply_path: str,
    cluster_ids_path: str,
    registry_path: str,
    out_graph_json: str,
    neighbor_dist_thresh: float = 0.02,
    min_points_per_cluster: int = 10,
):
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Missing ply: {ply_path}")
    if not os.path.exists(cluster_ids_path):
        raise FileNotFoundError(f"Missing cluster ids npy: {cluster_ids_path}")
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Missing registry: {registry_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    cluster_ids = np.load(cluster_ids_path).reshape(-1)

    if cluster_ids.shape[0] != points.shape[0]:
        raise RuntimeError(f"cluster_ids length {cluster_ids.shape[0]} != point count {points.shape[0]}")

    registry = _load_registry(registry_path)
    cluster_to_indices = _cluster_points(points, cluster_ids)

    eligible = []
    for cid, idxs in cluster_to_indices.items():
        info = registry.get(cid, None)
        if info is None:
            continue
        lab = str(info.get("label", "unknown"))
        if _is_unknown_label(lab):
            continue
        if idxs.shape[0] < min_points_per_cluster:
            continue
        pts = points[idxs]
        if _is_degenerate_cluster_points(pts):
            continue
        eligible.append(cid)
    eligible = sorted(eligible)

    aabbs = {}
    for cid in eligible:
        pts = points[cluster_to_indices[cid]]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < min_points_per_cluster:
            continue
        if _is_degenerate_cluster_points(pts):
            continue
        aabbs[cid] = _aabb_from_points(pts)

    eligible = [cid for cid in eligible if cid in aabbs]
    edges = []
    adj = {cid: [] for cid in eligible}

    for i in range(len(eligible)):
        cid_i = eligible[i]
        a_min, a_max = aabbs[cid_i]
        for j in range(i + 1, len(eligible)):
            cid_j = eligible[j]
            b_min, b_max = aabbs[cid_j]
            d = _aabb_min_distance(a_min, a_max, b_min, b_max)
            if d <= neighbor_dist_thresh:
                adj[cid_i].append(cid_j)
                adj[cid_j].append(cid_i)
                edges.append({"a": int(cid_i), "b": int(cid_j), "approx_aabb_dist": float(d)})

    graph = {
        "neighbor_dist_thresh": float(neighbor_dist_thresh),
        "min_points_per_cluster": int(min_points_per_cluster),
        "distance_mode": "aabb_min_distance_approx",
        "nodes": [
            {
                "cluster_id": int(cid),
                "label": str(registry[cid].get("label", "unknown")),
                "point_count": int(cluster_to_indices[cid].shape[0]),
                "aabb": {
                    "min": aabbs[cid][0].tolist(),
                    "max": aabbs[cid][1].tolist(),
                },
            }
            for cid in eligible
        ],
        "edges": edges,
        "adjacency": {str(cid): [int(x) for x in adj[cid]] for cid in eligible},
    }

    os.makedirs(os.path.dirname(out_graph_json), exist_ok=True)
    with open(out_graph_json, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"[GRAPH] Saved neighbor graph: {out_graph_json}")
    print(f"[GRAPH] Nodes kept (non-unknown*, non-degenerate): {len(graph['nodes'])}")
    return graph


# -----------------------------------------------------------------------------
# Merging + Copy unknowns
# -----------------------------------------------------------------------------
def merge_neighboring_clusters_same_label(
    ply_path: str,
    cluster_ids_path: str,
    registry_path: str,
    primitives_json_path: str,  # kept for compatibility
    out_dir: str,
    neighbor_dist_thresh: float = 0.02,
    min_points_per_cluster: int = 10,
    min_points_after_merge: int = 10,
    prune_disconnected_instances_keep_largest: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    cluster_ids = np.load(cluster_ids_path).reshape(-1)

    if cluster_ids.shape[0] != points.shape[0]:
        raise RuntimeError("cluster_ids.npy must align 1:1 with the PLY point order")

    registry = _load_registry(registry_path)
    cluster_to_indices = _cluster_points(points, cluster_ids)

    # known-only graph
    graph_path = os.path.join(out_dir, "neighbor_graph.json")
    graph = build_neighbor_graph(
        ply_path=ply_path,
        cluster_ids_path=cluster_ids_path,
        registry_path=registry_path,
        out_graph_json=graph_path,
        neighbor_dist_thresh=neighbor_dist_thresh,
        min_points_per_cluster=min_points_per_cluster,
    )
    adjacency = {int(k): [int(x) for x in v] for k, v in graph["adjacency"].items()}

    # group by known label
    label_to_cids = defaultdict(list)
    for node in graph["nodes"]:
        cid = int(node["cluster_id"])
        lab = str(node["label"])
        if _is_unknown_label(lab):
            continue
        label_to_cids[lab].append(cid)

    # merge by connectivity per label
    merged_groups = []
    visited = set()
    for lab, cids in label_to_cids.items():
        cids_set = set(cids)
        for start in cids:
            if start in visited:
                continue
            comp = []
            q = deque([start])
            visited.add(start)
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nb in adjacency.get(cur, []):
                    if nb in visited:
                        continue
                    if nb in cids_set:
                        visited.add(nb)
                        q.append(nb)
            merged_groups.append((lab, sorted(comp)))

    merged_cluster_ids = np.full_like(cluster_ids, -1, dtype=np.int32)
    merged_registry = {}
    merged_id = 0
    label_instance_counter = defaultdict(int)

    # --- known merges
    for lab, group in merged_groups:
        all_idxs_list = []
        for old_cid in group:
            idxs = cluster_to_indices.get(old_cid, None)
            if idxs is None:
                continue
            all_idxs_list.append(idxs)
        if not all_idxs_list:
            continue

        all_idxs = np.concatenate(all_idxs_list, axis=0)
        if all_idxs.shape[0] < min_points_after_merge:
            continue

        pts = points[all_idxs]
        if _is_degenerate_cluster_points(pts):
            continue  # <<< ignore single-point / degenerate cluster merges

        inst = label_instance_counter[lab]
        label_instance_counter[lab] += 1
        lab_out = f"{lab}_{inst}"

        merged_cluster_ids[all_idxs] = merged_id
        merged_registry[str(merged_id)] = {
            "label": lab_out,
            "members": [int(x) for x in group],
            "point_count": int(all_idxs.shape[0]),
            "source": "merged_known",
        }
        merged_id += 1

    # --- prune known disconnected instances (largest per base label)
    if prune_disconnected_instances_keep_largest and merged_registry:
        base_to_mids = defaultdict(list)
        for mid_str, info in merged_registry.items():
            lab = str(info.get("label", "unknown"))
            if _is_unknown_label(lab):
                continue
            base_to_mids[_base_label(lab)].append(int(mid_str))

        mids_to_delete = set()
        for base, mids in base_to_mids.items():
            if len(mids) <= 1:
                continue
            best_mid = max(mids, key=lambda m: int(merged_registry[str(m)].get("point_count", 0)))
            for m in mids:
                if m != best_mid:
                    mids_to_delete.add(m)

        if mids_to_delete:
            for m in mids_to_delete:
                merged_cluster_ids[merged_cluster_ids == m] = -1
            for m in sorted(mids_to_delete):
                merged_registry.pop(str(m), None)

            print(
                f"[PRUNE] pruned {len(mids_to_delete)} disconnected known instances (kept largest per base label)"
            )

    # --- copy unknowns (no merge) BUT skip degenerate/single-point clusters
    unknown_cids = []
    for cid, idxs in cluster_to_indices.items():
        info = registry.get(cid, None)
        if info is None:
            continue
        lab = str(info.get("label", "unknown"))
        if not _is_unknown_label(lab):
            continue
        if idxs.shape[0] < min_points_per_cluster:
            continue
        pts = points[idxs]
        if _is_degenerate_cluster_points(pts):
            continue  # <<< ignore single-point / degenerate unknown clusters
        unknown_cids.append(cid)
    unknown_cids = sorted(unknown_cids)

    unknown_counter = 0
    for cid in unknown_cids:
        idxs = cluster_to_indices[cid]
        lab_out = f"unknown_{unknown_counter}"
        unknown_counter += 1

        merged_cluster_ids[idxs] = merged_id
        merged_registry[str(merged_id)] = {
            "label": lab_out,
            "members": [int(cid)],
            "point_count": int(idxs.shape[0]),
            "source": "copied_unknown",
            "original_registry_label": str(registry[cid].get("label", "unknown")),
        }
        merged_id += 1

    print(f"[UNKNOWN] Copied unknown clusters (non-degenerate): {unknown_counter}")

    # --- build primitives for kept clusters, skipping degenerate again (safety)
    merged_primitives = []
    for mid_str, info in merged_registry.items():
        mid = int(mid_str)
        idxs = np.where(merged_cluster_ids == mid)[0]
        if idxs.shape[0] == 0:
            continue
        pts_mid = points[idxs]
        if _is_degenerate_cluster_points(pts_mid):
            continue

        params = _safe_obb_or_aabb_params(pts_mid)
        if params is None:
            continue

        merged_primitives.append(
            {
                "cluster_id": mid,
                "label": info["label"],
                "members": info.get("members", []),
                "parameters": {
                    "center": params["center"],
                    "extent": params["extent"],
                    "rotation": params["rotation"],
                    "bbox_mode": params["bbox_mode"],
                },
                "point_count": int(idxs.shape[0]),
                "source": info.get("source", "unknown"),
            }
        )

    # --- save merged ids
    merged_ids_path = os.path.join(out_dir, "merged_cluster_ids.npy")
    np.save(merged_ids_path, merged_cluster_ids)
    print(f"[SAVE] {merged_ids_path}")

    # --- save mapping (includes unknowns, excludes degenerate ones by construction)
    merged_map_path = os.path.join(out_dir, "merged_cluster_to_label.json")
    with open(merged_map_path, "w") as f:
        json.dump(merged_registry, f, indent=2)
    print(f"[SAVE] {merged_map_path}")

    # --- save primitives (only non-degenerate)
    merged_primitives_path = os.path.join(out_dir, "merged_pca_primitives.json")
    with open(merged_primitives_path, "w") as f:
        json.dump(
            {
                "source_ply": os.path.abspath(ply_path),
                "source_cluster_ids": os.path.abspath(cluster_ids_path),
                "source_registry": os.path.abspath(registry_path),
                "neighbor_dist_thresh": float(neighbor_dist_thresh),
                "distance_mode": "aabb_min_distance_approx",
                "primitives": merged_primitives,
            },
            f,
            indent=2,
        )
    print(f"[SAVE] {merged_primitives_path}")

    # --- colored PLY
    def _label_to_rgb01(label: str):
        import colorsys
        h = abs(hash(label)) % 360
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.8, 1.0)
        return np.array([r, g, b], dtype=np.float64)

    out_colors = np.zeros((points.shape[0], 3), dtype=np.float64) + 0.1  # unassigned = dark gray
    for mid_str, info in merged_registry.items():
        mid = int(mid_str)
        out_colors[merged_cluster_ids == mid] = _label_to_rgb01(info["label"])

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(points)
    out_pcd.colors = o3d.utility.Vector3dVector(out_colors)

    merged_ply_path = os.path.join(out_dir, "merged_labeled_clusters.ply")
    o3d.io.write_point_cloud(merged_ply_path, out_pcd)
    print(f"[SAVE] {merged_ply_path}")

    return {
        "neighbor_graph_json": graph_path,
        "merged_cluster_ids_npy": merged_ids_path,
        "merged_cluster_to_label_json": merged_map_path,
        "merged_primitives_json": merged_primitives_path,
        "merged_ply": merged_ply_path,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cluster_ids", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--primitives", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--neighbor_dist", type=float, default=0.02)
    ap.add_argument("--min_points", type=int, default=10)
    ap.add_argument("--min_points_after_merge", type=int, default=10)
    ap.add_argument("--no_prune_keep_largest", action="store_true")
    args = ap.parse_args()

    merge_neighboring_clusters_same_label(
        ply_path=args.ply,
        cluster_ids_path=args.cluster_ids,
        registry_path=args.registry,
        primitives_json_path=args.primitives,
        out_dir=args.out_dir,
        neighbor_dist_thresh=args.neighbor_dist,
        min_points_per_cluster=args.min_points,
        min_points_after_merge=args.min_points_after_merge,
        prune_disconnected_instances_keep_largest=(not args.no_prune_keep_largest),
    )


if __name__ == "__main__":
    main()
