#!/usr/bin/env python3
"""
vis_cluster_label_assignment.py

Inputs:
- sketch/3d_reconstruction/clustering_k20_points.npy   (cluster id per point)
- sketch/3d_reconstruction/clustering_k20_points.ply   (points)
- sketch/partfield_overlay/merged_label_ids.npy        (semantic label id per point, -1 = unlabeled)
- sketch/partfield_overlay/label_color_map.json        (label names + colors)

Logic:
For each cluster:
- find majority semantic label among points with sem_id>=0
- occupation = majority_count / total_points_in_cluster
- if occupation > 0.50 => assign that label
- else => assign unknown_k (one unknown per such cluster)

Visualization:
- For each known label: show all points whose cluster is assigned to that label (colored), others black.
- Then show all unknown clusters together (colored), others black.

Also prints:
- cluster -> assigned label (and stats)
- label -> clusters
"""

import os
import json
import numpy as np
import open3d as o3d

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")

CLUSTER_NPY = os.path.join(SKETCH_ROOT, "3d_reconstruction", "clustering_k20_points.npy")
CLUSTER_PLY = os.path.join(SKETCH_ROOT, "3d_reconstruction", "clustering_k20_points.ply")

OVERLAY_DIR = os.path.join(SKETCH_ROOT, "partfield_overlay")
SEM_NPY = os.path.join(OVERLAY_DIR, "merged_label_ids.npy")
LABEL_COLOR_JSON = os.path.join(OVERLAY_DIR, "label_color_map.json")

THRESH = 0.50  # > 50%


def load_points(ply_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise RuntimeError(f"Bad/empty point cloud: {ply_path}")
    return pts


def distinct_colors_rgb01(n: int) -> np.ndarray:
    """Deterministic distinct-ish colors for unknowns, (n,3) in [0,1]."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    cols = []
    for i in range(n):
        h = (i / max(1, n)) * 6.0
        c = 1.0
        x = c * (1.0 - abs((h % 2.0) - 1.0))
        if 0 <= h < 1:
            rgb = (c, x, 0)
        elif 1 <= h < 2:
            rgb = (x, c, 0)
        elif 2 <= h < 3:
            rgb = (0, c, x)
        elif 3 <= h < 4:
            rgb = (0, x, c)
        elif 4 <= h < 5:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
        rgb = np.array(rgb, dtype=np.float64)
        rgb = 0.25 + 0.75 * rgb
        cols.append(np.clip(rgb, 0.0, 1.0))
    return np.stack(cols, axis=0)


def main():
    # ---- checks ----
    for p in [CLUSTER_NPY, CLUSTER_PLY, SEM_NPY, LABEL_COLOR_JSON]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing: {p}")

    clusters = np.load(CLUSTER_NPY).reshape(-1).astype(np.int32)
    sem_ids = np.load(SEM_NPY).reshape(-1).astype(np.int32)
    if clusters.shape[0] != sem_ids.shape[0]:
        raise ValueError(f"Length mismatch: clusters={clusters.shape[0]} vs sem_ids={sem_ids.shape[0]}")

    pts = load_points(CLUSTER_PLY)
    if pts.shape[0] != clusters.shape[0]:
        raise ValueError(f"Point count mismatch: ply_points={pts.shape[0]} vs clusters={clusters.shape[0]}")

    with open(LABEL_COLOR_JSON, "r") as f:
        lmap = json.load(f)
    labels_in_order = lmap["labels_in_order"]
    label_to_color_bgr = lmap["label_to_color_bgr"]  # label -> [b,g,r]

    # known label colors in RGB01 aligned to labels_in_order
    known_rgb01 = np.zeros((len(labels_in_order), 3), dtype=np.float64)
    for i, lab in enumerate(labels_in_order):
        b, g, r = label_to_color_bgr[lab]
        known_rgb01[i] = np.array([r, g, b], dtype=np.float64) / 255.0

    # ---- compute cluster -> assigned label (or unknown) ----
    cluster_ids = np.unique(clusters).tolist()

    cluster_to_assigned = {}   # cid -> name
    cluster_stats = {}         # cid -> stats
    unlabeled_clusters = []    # cids that become unknown_k

    for cid in cluster_ids:
        idx = np.where(clusters == cid)[0]
        n_cluster = int(idx.size)
        if n_cluster == 0:
            continue

        sem_in = sem_ids[idx]
        valid_sem = sem_in[sem_in >= 0]

        if valid_sem.size == 0:
            unlabeled_clusters.append(int(cid))
            cluster_stats[int(cid)] = {
                "n_points": n_cluster,
                "majority_label": None,
                "majority_count": 0,
                "majority_frac_of_cluster": 0.0,
                "note": "no sem_id>=0 points in this cluster",
            }
            continue

        vals, counts = np.unique(valid_sem, return_counts=True)
        j = int(np.argmax(counts))
        maj_sem_id = int(vals[j])
        maj_count = int(counts[j])
        maj_frac = float(maj_count) / float(n_cluster)  # fraction of ALL points in cluster

        if maj_frac > THRESH:
            name = labels_in_order[maj_sem_id]
            cluster_to_assigned[int(cid)] = name
        else:
            unlabeled_clusters.append(int(cid))

        cluster_stats[int(cid)] = {
            "n_points": n_cluster,
            "majority_label_id": maj_sem_id,
            "majority_label": labels_in_order[maj_sem_id],
            "majority_count": maj_count,
            "majority_frac_of_cluster": maj_frac,
            "n_semantic_labeled_points_in_cluster": int(valid_sem.size),
        }

    unlabeled_clusters = sorted(unlabeled_clusters)
    unknown_name_by_cluster = {cid: f"unknown_{k}" for k, cid in enumerate(unlabeled_clusters)}
    for cid, uname in unknown_name_by_cluster.items():
        cluster_to_assigned[int(cid)] = uname

    # ---- invert mapping: label -> clusters ----
    label_to_clusters = {}
    for cid, name in cluster_to_assigned.items():
        label_to_clusters.setdefault(name, []).append(int(cid))
    for name in label_to_clusters:
        label_to_clusters[name] = sorted(label_to_clusters[name])

    # ---- print cluster -> label ----
    print("\n=== cluster -> assigned label ===")
    for cid in sorted(cluster_to_assigned.keys()):
        name = cluster_to_assigned[cid]
        st = cluster_stats.get(cid, {})
        frac = st.get("majority_frac_of_cluster", None)
        maj = st.get("majority_label", None)
        if frac is None:
            print(f"cluster {cid:>3} -> {name}")
        else:
            # show both: assigned and the winning candidate + fraction
            print(f"cluster {cid:>3} -> {name:>12}   (winner={maj}, frac={frac:.3f}, n={st.get('n_points',0)})")

    # ---- print label -> clusters (known first, then unknowns) ----
    print("\n=== label -> clusters ===")
    for lab in labels_in_order:
        cids = label_to_clusters.get(lab, [])
        if cids:
            print(f"{lab}: {cids}")
    unk_key_list = [f"unknown_{k}" for k in range(len(unlabeled_clusters))]
    if unk_key_list:
        print("\nUnknowns:")
        for uk in unk_key_list:
            print(f"{uk}: {label_to_clusters.get(uk, [])}")

    # ---- visualization helpers ----
    unknown_colors = distinct_colors_rgb01(len(unlabeled_clusters))

    def make_pcd(mask: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
        """mask selects points to color; others black."""
        cols = np.zeros((pts.shape[0], 3), dtype=np.float64)
        cols[mask] = colors[mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        return pcd

    # Precompute per-point assigned name
    # (each point inherits from its cluster assignment)
    assigned_name_per_point = np.empty((pts.shape[0],), dtype=object)
    for cid in cluster_ids:
        name = cluster_to_assigned.get(int(cid), "unknown_unseen")
        assigned_name_per_point[clusters == cid] = name

    # ---- show each known label once ----
    for lab_id, lab in enumerate(labels_in_order):
        cids = label_to_clusters.get(lab, [])
        if not cids:
            continue

        mask = np.isin(clusters, np.array(cids, dtype=np.int32))
        # color all selected points with that label's color
        colors = np.zeros((pts.shape[0], 3), dtype=np.float64)
        colors[mask] = known_rgb01[lab_id]

        pcd = make_pcd(mask, colors)
        title = f"Label: {lab}  (clusters={cids})"
        print(f"\n[VIS] {title}")
        o3d.visualization.draw_geometries([pcd], window_name=title)

    # ---- show all unknowns together ----
    if unlabeled_clusters:
        mask_any_unk = np.isin(clusters, np.array(unlabeled_clusters, dtype=np.int32))
        colors = np.zeros((pts.shape[0], 3), dtype=np.float64)

        # each unknown cluster gets its own color, but all shown in one view
        for k, cid in enumerate(unlabeled_clusters):
            m = (clusters == cid)
            colors[m] = unknown_colors[k]

        pcd = make_pcd(mask_any_unk, colors)
        title = f"Unknown clusters together (count={len(unlabeled_clusters)})"
        print(f"\n[VIS] {title}")
        o3d.visualization.draw_geometries([pcd], window_name=title)
    else:
        print("\n[VIS] No unknown clusters (all clusters assigned to known labels).")


if __name__ == "__main__":
    main()
