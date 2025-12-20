#!/usr/bin/env python3
import os
import json
import numpy as np
import open3d as o3d
from collections import defaultdict, deque

try:
    from scipy.spatial import cKDTree
except Exception as e:
    cKDTree = None


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
        except:
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


def _min_intercluster_distance(points_a: np.ndarray, points_b: np.ndarray):
    """
    Compute approximate min distance between two point sets using KDTree.
    Uses the smaller set to query the larger set's tree.
    """
    if cKDTree is None:
        raise RuntimeError("scipy is required (cKDTree). Please: pip install scipy")

    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return float("inf")

    # query from smaller to larger
    if points_a.shape[0] <= points_b.shape[0]:
        small, large = points_a, points_b
    else:
        small, large = points_b, points_a

    tree = cKDTree(large)
    dists, _ = tree.query(small, k=1, workers=-1)
    return float(np.min(dists))


def build_neighbor_graph(
    ply_path: str,
    cluster_ids_path: str,
    registry_path: str,
    out_graph_json: str,
    neighbor_dist_thresh: float = 0.02,
    min_points_per_cluster: int = 10,
):
    """
    Build adjacency graph between clusters based on min point-to-point distance.
    Only includes clusters that appear in registry and have enough points.
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
        raise RuntimeError(
            f"cluster_ids length {cluster_ids.shape[0]} != point count {points.shape[0]}"
        )

    registry = _load_registry(registry_path)
    cluster_to_indices = _cluster_points(points, cluster_ids)

    # eligible clusters: in registry + have enough points
    eligible = []
    for cid, idxs in cluster_to_indices.items():
        if cid not in registry:
            continue
        if idxs.shape[0] < min_points_per_cluster:
            continue
        eligible.append(cid)
    eligible = sorted(eligible)

    edges = []
    adj = {cid: [] for cid in eligible}

    # pairwise check (20-40 clusters -> fine)
    for i in range(len(eligible)):
        cid_i = eligible[i]
        pts_i = points[cluster_to_indices[cid_i]]
        for j in range(i + 1, len(eligible)):
            cid_j = eligible[j]
            pts_j = points[cluster_to_indices[cid_j]]

            d = _min_intercluster_distance(pts_i, pts_j)
            if d <= neighbor_dist_thresh:
                adj[cid_i].append(cid_j)
                adj[cid_j].append(cid_i)
                edges.append({"a": cid_i, "b": cid_j, "min_dist": d})

    graph = {
        "neighbor_dist_thresh": float(neighbor_dist_thresh),
        "min_points_per_cluster": int(min_points_per_cluster),
        "nodes": [
            {
                "cluster_id": int(cid),
                "label": str(registry[cid].get("label", "unknown")),
                "point_count": int(cluster_to_indices[cid].shape[0]),
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
    1) Build neighbor graph
    2) Merge connected components where all nodes share the same label (and are connected through neighbor edges)
    3) Produce:
       - merged_cluster_ids.npy   (per-point merged cluster id, aligned to ply)
       - merged_labeled_clusters.ply (same geometry colored by label)
       - merged_cluster_to_label.json (new merged id -> label + members)
       - merged_pca_primitives.json (OBB recomputed per merged cluster)
       - neighbor_graph.json (debug)
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- load base data
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    cluster_ids = np.load(cluster_ids_path).reshape(-1)
    if cluster_ids.shape[0] != points.shape[0]:
        raise RuntimeError("cluster_ids.npy must align 1:1 with the PLY point order")

    registry = _load_registry(registry_path)

    # --- build neighbor graph
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
    merged_groups = []  # list of lists of old cluster ids
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

    # clusters that are eligible but unknown or isolated label groups still appear as singletons
    # (unknown clusters are left as -1 in merged ids)
    # Note: if you want to keep unknown clusters as their own groups later, we can add that.

    # --- construct per-point merged ids
    cluster_to_indices = _cluster_points(points, cluster_ids)
    merged_cluster_ids = np.full_like(cluster_ids, -1, dtype=np.int32)

    merged_registry = {}
    merged_id = 0

    for lab, group in merged_groups:
        # gather point indices
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

        merged_cluster_ids[all_idxs] = merged_id
        merged_registry[str(merged_id)] = {
            "label": lab,
            "members": [int(x) for x in group],
            "point_count": int(all_idxs.shape[0]),
        }
        merged_id += 1

    # --- save merged ids
    merged_ids_path = os.path.join(out_dir, "merged_cluster_ids.npy")
    np.save(merged_ids_path, merged_cluster_ids)
    print(f"[MERGE] Saved: {merged_ids_path} (merged clusters: {merged_id})")

    # --- save merged cluster_to_label mapping json
    merged_map_path = os.path.join(out_dir, "merged_cluster_to_label.json")
    with open(merged_map_path, "w") as f:
        json.dump(merged_registry, f, indent=2)
    print(f"[MERGE] Saved: {merged_map_path}")

    # --- recompute PCA OBB for merged clusters (using Open3D OBB)
    merged_primitives = []
    for mid_str, info in merged_registry.items():
        mid = int(mid_str)
        idxs = np.where(merged_cluster_ids == mid)[0]
        if idxs.shape[0] < min_points_after_merge:
            continue

        temp = o3d.geometry.PointCloud()
        temp.points = o3d.utility.Vector3dVector(points[idxs])
        obb = temp.get_oriented_bounding_box()

        merged_primitives.append({
            "cluster_id": mid,
            "label": info["label"],
            "members": info["members"],
            "parameters": {
                "center": obb.center.tolist(),
                "extent": obb.extent.tolist(),
                "rotation": obb.R.tolist(),
            },
            "point_count": int(idxs.shape[0]),
        })

    merged_primitives_path = os.path.join(out_dir, "merged_pca_primitives.json")
    with open(merged_primitives_path, "w") as f:
        json.dump(
            {
                "source_ply": os.path.abspath(ply_path),
                "source_cluster_ids": os.path.abspath(cluster_ids_path),
                "neighbor_dist_thresh": float(neighbor_dist_thresh),
                "primitives": merged_primitives,
            },
            f,
            indent=2,
        )
    print(f"[MERGE] Saved: {merged_primitives_path}")

    # --- write a merged visualization PLY (color by semantic label)
    # If your registry from cluster_map had colors, you can reuse them.
    # Here we just generate stable colors by label hashing for now.
    def _label_to_rgb01(label: str):
        # deterministic pseudo-color
        h = abs(hash(label)) % 360
        # simple hsv->rgb
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
