#!/usr/bin/env python3
import os
import json
import numpy as np
import open3d as o3d
from collections import defaultdict, deque


def _load_registry(registry_path: str):
    """
    registry.json produced by build_registry_from_cluster_map().
    Expected keys: "0","1","2"... (cluster_id as string), with fields like:
      {"label": "...", "color_rgb": [r,g,b] (optional)}
    """
    with open(registry_path, "r") as f:
        reg = json.load(f)

    # normalize keys to int
    reg_int = {}
    for k, v in reg.items():
        try:
            cid = int(k)
        except Exception:
            continue
        reg_int[cid] = v
    return reg_int


def _cluster_points(points: np.ndarray, cluster_ids: np.ndarray):
    """
    Returns dict cluster_id -> np.ndarray point indices
    """
    clusters = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        if cid < 0:
            continue
        clusters[int(cid)].append(i)
    return {cid: np.array(idxs, dtype=np.int64) for cid, idxs in clusters.items()}


def _aabb_from_points(pts: np.ndarray):
    """
    Returns (mn, mx) each shape (3,)
    """
    mn = np.min(pts, axis=0)
    mx = np.max(pts, axis=0)
    return mn, mx


def _aabb_min_distance(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    """
    Euclidean distance between two axis-aligned bounding boxes in 3D.
    If they intersect, distance = 0.
    """
    # gap per axis: if overlapping => 0
    dx = max(0.0, float(b_min[0] - a_max[0]), float(a_min[0] - b_max[0]))
    dy = max(0.0, float(b_min[1] - a_max[1]), float(a_min[1] - b_max[1]))
    dz = max(0.0, float(b_min[2] - a_max[2]), float(a_min[2] - b_max[2]))
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def build_neighbor_graph(
    ply_path: str,
    cluster_ids_path: str,
    registry_path: str,
    out_graph_json: str,
    neighbor_dist_thresh: float = 0.02,
    min_points_per_cluster: int = 10,
):
    """
    Build an APPROXIMATE adjacency graph between clusters based on AABB-to-AABB distance.
    This is fast and sufficient for merging.

    Only includes clusters that:
      - appear in registry
      - have >= min_points_per_cluster points
    """
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

    # eligible clusters
    eligible = []
    for cid, idxs in cluster_to_indices.items():
        if cid not in registry:
            continue
        if idxs.shape[0] < min_points_per_cluster:
            continue
        eligible.append(cid)
    eligible = sorted(eligible)

    # precompute AABBs once (cluster-level)
    aabbs = {}
    for cid in eligible:
        pts = points[cluster_to_indices[cid]]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < min_points_per_cluster:
            continue
        aabbs[cid] = _aabb_from_points(pts)

    eligible = [cid for cid in eligible if cid in aabbs]
    edges = []
    adj = {cid: [] for cid in eligible}

    # pairwise AABB distance (very fast)
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
    return graph


def merge_neighboring_clusters_same_label(
    ply_path: str,
    cluster_ids_path: str,
    registry_path: str,
    primitives_json_path: str,
    out_dir: str,
    neighbor_dist_thresh: float = 0.02,
    min_points_per_cluster: int = 10,
    min_points_after_merge: int = 10,
):
    """
    1) Build neighbor graph (FAST APPROX via AABBs)
    2) Merge connected components where all nodes share the same label (and are connected through neighbor edges)
    3) Produce:
       - merged_cluster_ids.npy   (per-point merged cluster id, aligned to ply)
       - merged_labeled_clusters.ply (same geometry colored by label)
       - merged_cluster_to_label.json (new merged id -> label + members)
       - merged_pca_primitives.json (OBB recomputed per merged cluster)
       - neighbor_graph.json (debug)

    Existing behavior kept:
      - disconnected components of same label become x_0, x_1, ...

    Heuristic A (kept):
      - for each base label x, keep only the largest instance (by point_count),
        delete all other disconnected instances entirely.

    Additional behavior (per your latest request):
      - if a merged cluster is invalid/degenerate (OBB fails / rank<3), remove it entirely.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- load base data
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    cluster_ids = np.load(cluster_ids_path).reshape(-1)
    if cluster_ids.shape[0] != points.shape[0]:
        raise RuntimeError("cluster_ids.npy must align 1:1 with the PLY point order")

    registry = _load_registry(registry_path)

    # --- build neighbor graph (fast)
    graph_path = os.path.join(out_dir, "neighbor_graph.json")
    graph = build_neighbor_graph(
        ply_path=ply_path,
        cluster_ids_path=cluster_ids_path,
        registry_path=registry_path,
        out_graph_json=graph_path,
        neighbor_dist_thresh=neighbor_dist_thresh,
        min_points_per_cluster=min_points_per_cluster,
    )
    adjacency = {int(k): v for k, v in graph["adjacency"].items()}

    # --- group by label first
    label_to_cids = defaultdict(list)
    for node in graph["nodes"]:
        cid = int(node["cluster_id"])
        lab = str(node["label"])
        if lab == "unknown":
            continue
        label_to_cids[lab].append(cid)

    # --- within each label, merge by connectivity in adjacency graph
    merged_groups = []  # list of (label, [old_cluster_ids])
    visited = set()

    for lab, cids in label_to_cids.items():
        cids_set = set(cids)

        for start in cids:
            if start in visited:
                continue

            # BFS within same-label subgraph
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

    # --- construct per-point merged ids
    cluster_to_indices = _cluster_points(points, cluster_ids)
    merged_cluster_ids = np.full_like(cluster_ids, -1, dtype=np.int32)

    merged_registry = {}
    merged_id = 0

    # disconnected components become x_0, x_1, ...
    label_instance_counter = defaultdict(int)

    for lab, group in merged_groups:
        all_idxs = []
        for old_cid in group:
            idxs = cluster_to_indices.get(old_cid, None)
            if idxs is None:
                continue
            all_idxs.append(idxs)

        if not all_idxs:
            continue

        all_idxs = np.concatenate(all_idxs, axis=0)
        if all_idxs.shape[0] < min_points_after_merge:
            continue

        inst = label_instance_counter[lab]
        label_instance_counter[lab] += 1
        lab_out = f"{lab}_{inst}"

        merged_cluster_ids[all_idxs] = merged_id
        merged_registry[str(merged_id)] = {
            "label": lab_out,
            "members": [int(x) for x in group],
            "point_count": int(all_idxs.shape[0]),
        }
        merged_id += 1

    # -------------------------------------------------------------------------
    # Heuristic A — keep only the largest instance per base label
    # -------------------------------------------------------------------------
    def _base_label_from_instance(label_with_suffix: str) -> str:
        """
        Expected: "x_0", "x_1", ...
        Only strips the last "_<int>" if it exists.
        """
        s = str(label_with_suffix)
        if "_" not in s:
            return s
        base, tail = s.rsplit("_", 1)
        try:
            int(tail)
            return base
        except Exception:
            return s

    base_to_mids = defaultdict(list)
    for mid_str, info in merged_registry.items():
        base = _base_label_from_instance(info.get("label", "unknown"))
        base_to_mids[base].append(int(mid_str))

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
            f"[PRUNE] Heuristic A: pruned {len(mids_to_delete)} disconnected instances "
            f"(kept largest per base label)"
        )

    # -------------------------------------------------------------------------
    # Drop invalid merged clusters (degenerate / OBB failure)
    # -------------------------------------------------------------------------
    def _is_invalid_cluster_points(pts: np.ndarray) -> bool:
        if pts is None or pts.shape[0] < min_points_after_merge:
            return True

        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < min_points_after_merge:
            return True

        pts_u = np.unique(np.round(pts, 6), axis=0)
        if pts_u.shape[0] < 4:
            return True

        X = pts_u - pts_u.mean(axis=0, keepdims=True)
        s = np.linalg.svd(X, compute_uv=False)
        rank = int((s > 1e-9).sum())
        return rank < 3

    invalid_mids = []

    merged_primitives = []
    for mid_str, info in list(merged_registry.items()):
        mid = int(mid_str)
        idxs = np.where(merged_cluster_ids == mid)[0]
        if idxs.shape[0] < min_points_after_merge:
            invalid_mids.append(mid)
            continue

        pts_mid = points[idxs]
        if _is_invalid_cluster_points(pts_mid):
            invalid_mids.append(mid)
            continue

        temp = o3d.geometry.PointCloud()
        temp.points = o3d.utility.Vector3dVector(pts_mid)

        try:
            obb = temp.get_oriented_bounding_box()
        except Exception:
            invalid_mids.append(mid)
            continue

        ext = np.array(obb.extent, dtype=np.float64)
        if not np.isfinite(ext).all() or np.min(ext) < 1e-8:
            invalid_mids.append(mid)
            continue

        merged_primitives.append(
            {
                "cluster_id": mid,
                "label": info["label"],
                "members": info["members"],
                "parameters": {
                    "center": obb.center.tolist(),
                    "extent": obb.extent.tolist(),
                    "rotation": obb.R.tolist(),
                },
                "point_count": int(idxs.shape[0]),
            }
        )

    if invalid_mids:
        for mid in invalid_mids:
            merged_cluster_ids[merged_cluster_ids == mid] = -1
            merged_registry.pop(str(mid), None)
        print(f"[PRUNE] Dropped {len(invalid_mids)} invalid merged clusters (degenerate / OBB-fail).")

    # --- save merged ids
    merged_ids_path = os.path.join(out_dir, "merged_cluster_ids.npy")
    np.save(merged_ids_path, merged_cluster_ids)
    print(f"[MERGE] Saved: {merged_ids_path} (kept merged clusters: {len(merged_registry)})")

    # --- save merged cluster_to_label mapping json
    merged_map_path = os.path.join(out_dir, "merged_cluster_to_label.json")
    with open(merged_map_path, "w") as f:
        json.dump(merged_registry, f, indent=2)
    print(f"[MERGE] Saved: {merged_map_path}")

    # --- save merged primitives json
    merged_primitives_path = os.path.join(out_dir, "merged_pca_primitives.json")
    with open(merged_primitives_path, "w") as f:
        json.dump(
            {
                "source_ply": os.path.abspath(ply_path),
                "source_cluster_ids": os.path.abspath(cluster_ids_path),
                "neighbor_dist_thresh": float(neighbor_dist_thresh),
                "distance_mode": "aabb_min_distance_approx",
                "primitives": merged_primitives,
            },
            f,
            indent=2,
        )
    print(f"[MERGE] Saved: {merged_primitives_path}")

    # --- write a merged visualization PLY (color by semantic label)
    def _label_to_rgb01(label: str):
        h = abs(hash(label)) % 360
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.8, 1.0)
        return np.array([r, g, b], dtype=np.float64)

    out_colors = np.zeros((points.shape[0], 3), dtype=np.float64) + 0.1  # unknown = dark gray
    for mid_str, info in merged_registry.items():
        mid = int(mid_str)
        rgb = _label_to_rgb01(info["label"])
        out_colors[merged_cluster_ids == mid] = rgb

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(points)
    out_pcd.colors = o3d.utility.Vector3dVector(out_colors)

    merged_ply_path = os.path.join(out_dir, "merged_labeled_clusters.ply")
    o3d.io.write_point_cloud(merged_ply_path, out_pcd)
    print(f"[MERGE] Saved: {merged_ply_path}")

    return {
        "neighbor_graph_json": graph_path,
        "merged_cluster_ids_npy": merged_ids_path,
        "merged_cluster_to_label_json": merged_map_path,
        "merged_primitives_json": merged_primitives_path,
        "merged_ply": merged_ply_path,
    }
