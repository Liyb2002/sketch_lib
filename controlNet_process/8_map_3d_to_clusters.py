#!/usr/bin/env python3
"""
vis_cluster_label_assignment.py  (with "all labels must be used" rule)

Inputs:
- sketch/3d_reconstruction/clustering_k20_points.npy   (cluster id per point)
- sketch/3d_reconstruction/clustering_k20_points.ply   (points)
- sketch/partfield_overlay/merged_label_ids.npy        (semantic label id per point, -1 = unlabeled)
- sketch/partfield_overlay/label_color_map.json        (label names + colors)

Rules:
A) Base assignment:
   For each cluster:
   - find majority semantic label among sem_id>=0 points
   - occupation = majority_count / total_points_in_cluster
   - if occupation > 0.50 => assign that label
   - else => unknown_k (one unknown per such cluster)

B) Additional rule (requested):
   Every known label must be used at least once.
   If a label has no cluster assigned after (A), we assign it the cluster where that label
   has the highest occupation (fraction of cluster points).
   Preference order when picking a cluster for a missing label:
     1) unknown clusters first
     2) clusters belonging to labels that have >=2 clusters (can spare one)
     3) any cluster (last resort)

Visualization:
- For each known label: show all clusters assigned to that label (colored), others black.
- For unknowns: show them together (each unknown cluster distinct color), others black.

Print:
- cluster -> assigned label (with stats)
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

THRESH = 0.50  # majority fraction must be > 0.50
VIS = False

def load_points(ply_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise RuntimeError(f"Bad/empty point cloud: {ply_path}")
    return pts


def distinct_colors_rgb01(n: int) -> np.ndarray:
    """Deterministic distinct-ish colors for unknown clusters, (n,3) in [0,1]."""
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


def invert_mapping(cluster_to_assigned: dict[int, str]) -> dict[str, list[int]]:
    label_to_clusters = {}
    for cid, name in cluster_to_assigned.items():
        label_to_clusters.setdefault(name, []).append(int(cid))
    for k in label_to_clusters:
        label_to_clusters[k] = sorted(label_to_clusters[k])
    return label_to_clusters


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

    cluster_ids = np.unique(clusters).tolist()

    # Precompute per-cluster sizes
    cluster_sizes = {int(cid): int((clusters == cid).sum()) for cid in cluster_ids}

    # Precompute occupation matrix (cluster -> label -> count and frac)
    # occ_frac[(cid, lid)] = (# points in cluster cid whose sem_id==lid) / (size of cluster cid)
    occ_count = {}  # (cid, lid) -> count
    occ_frac = {}   # (cid, lid) -> frac

    for cid in cluster_ids:
        cid = int(cid)
        idx = np.where(clusters == cid)[0]
        n_cluster = int(idx.size)
        if n_cluster == 0:
            continue
        sem_in = sem_ids[idx]
        for lid in np.unique(sem_in[sem_in >= 0]):
            lid = int(lid)
            c = int((sem_in == lid).sum())
            occ_count[(cid, lid)] = c
            occ_frac[(cid, lid)] = float(c) / float(n_cluster)

    # ---- A) Base assignment: majority > 0.5 -> label else unknown_k ----
    cluster_to_assigned: dict[int, str] = {}
    cluster_stats: dict[int, dict] = {}
    unknown_candidate_clusters: list[int] = []

    for cid in cluster_ids:
        cid = int(cid)
        idx = np.where(clusters == cid)[0]
        n_cluster = int(idx.size)
        if n_cluster == 0:
            continue

        sem_in = sem_ids[idx]
        valid_sem = sem_in[sem_in >= 0]

        if valid_sem.size == 0:
            unknown_candidate_clusters.append(cid)
            cluster_stats[cid] = {
                "n_points": n_cluster,
                "majority_label": None,
                "majority_count": 0,
                "majority_frac_of_cluster": 0.0,
                "note": "no sem_id>=0 points in this cluster",
            }
            continue

        vals, counts = np.unique(valid_sem, return_counts=True)
        j = int(np.argmax(counts))
        maj_lid = int(vals[j])
        maj_count = int(counts[j])
        maj_frac = float(maj_count) / float(n_cluster)

        cluster_stats[cid] = {
            "n_points": n_cluster,
            "majority_label_id": maj_lid,
            "majority_label": labels_in_order[maj_lid],
            "majority_count": maj_count,
            "majority_frac_of_cluster": maj_frac,
            "n_semantic_labeled_points_in_cluster": int(valid_sem.size),
        }

        if maj_frac > THRESH:
            cluster_to_assigned[cid] = labels_in_order[maj_lid]
        else:
            unknown_candidate_clusters.append(cid)

    unknown_candidate_clusters = sorted(unknown_candidate_clusters)

    # Assign unknown_0, unknown_1... to those unknown candidates (temporary; may get reassigned in pass B)
    unknown_name_by_cluster = {cid: f"unknown_{k}" for k, cid in enumerate(unknown_candidate_clusters)}
    for cid, uname in unknown_name_by_cluster.items():
        cluster_to_assigned[cid] = uname

    # ---- B) Ensure ALL known labels are used at least once ----
    # Find missing known labels
    label_to_clusters = invert_mapping(cluster_to_assigned)

    missing_labels = [lab for lab in labels_in_order if lab not in label_to_clusters or len(label_to_clusters[lab]) == 0]

    if missing_labels:
        print("\n[INFO] Missing labels (will force-assign):", missing_labels)

    # Helper: current cluster "owner" label
    def owner_label(cid: int) -> str:
        return cluster_to_assigned.get(cid, "")

    def is_unknown_name(name: str) -> bool:
        return name.startswith("unknown_")

    def pick_best_cluster_for_label(missing_lab: str) -> int | None:
        """Pick cluster with highest occupation for this label, using preference order."""
        lid = labels_in_order.index(missing_lab)

        # Compute best occupation among all clusters
        # We'll gather candidates with their frac
        best_any = None
        best_any_frac = -1.0

        best_unknown = None
        best_unknown_frac = -1.0

        best_stealable = None
        best_stealable_frac = -1.0

        # Recompute label_to_clusters each call to be accurate after reassignments
        current_label_to_clusters = invert_mapping(cluster_to_assigned)

        for cid in cluster_ids:
            cid = int(cid)
            frac = occ_frac.get((cid, lid), 0.0)
            if frac <= 0.0:
                continue

            # Any cluster
            if frac > best_any_frac:
                best_any_frac = frac
                best_any = cid

            # Unknown clusters preferred
            if is_unknown_name(owner_label(cid)):
                if frac > best_unknown_frac:
                    best_unknown_frac = frac
                    best_unknown = cid
                continue

            # Stealable: belongs to a label that has >=2 clusters
            owner = owner_label(cid)
            owner_clusters = current_label_to_clusters.get(owner, [])
            if len(owner_clusters) >= 2:
                if frac > best_stealable_frac:
                    best_stealable_frac = frac
                    best_stealable = cid

        if best_unknown is not None:
            return best_unknown
        if best_stealable is not None:
            return best_stealable
        return best_any  # last resort (may steal the only cluster from someone)

    # Force-assign each missing label
    for lab in missing_labels:
        cid = pick_best_cluster_for_label(lab)
        if cid is None:
            print(f"[WARN] Could not find any cluster with nonzero occupation for label '{lab}'. Skipping.")
            continue
        prev = cluster_to_assigned.get(cid, None)
        cluster_to_assigned[cid] = lab
        print(f"[FORCE] label '{lab}' assigned cluster {cid} (was '{prev}')")

    # Rebuild final label_to_clusters after forcing
    label_to_clusters = invert_mapping(cluster_to_assigned)

    # Rebuild unknown list (whatever is still unknown after forcing)
    unknown_keys = sorted([k for k in label_to_clusters.keys() if is_unknown_name(k)])
    unknown_clusters_final = []
    for uk in unknown_keys:
        unknown_clusters_final.extend(label_to_clusters.get(uk, []))
    unknown_clusters_final = sorted(set(unknown_clusters_final))

    # ---- Print cluster -> label ----
    print("\n=== cluster -> assigned label ===")
    for cid in sorted(cluster_to_assigned.keys()):
        name = cluster_to_assigned[cid]
        st = cluster_stats.get(cid, {})
        frac = st.get("majority_frac_of_cluster", None)
        maj = st.get("majority_label", None)
        if frac is None:
            print(f"cluster {cid:>3} -> {name}")
        else:
            print(f"cluster {cid:>3} -> {name:>12}   (winner={maj}, frac={frac:.3f}, n={st.get('n_points',0)})")

    # ---- Print label -> clusters ----
    print("\n=== label -> clusters ===")
    for lab in labels_in_order:
        cids = label_to_clusters.get(lab, [])
        print(f"{lab}: {cids}")

    if unknown_keys:
        print("\nUnknowns:")
        for uk in unknown_keys:
            print(f"{uk}: {label_to_clusters.get(uk, [])}")

    # ---- Visualization ----
    if VIS:
        unknown_colors = distinct_colors_rgb01(max(1, len(unknown_keys)))  # used per unknown group key

        def make_pcd(mask: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
            cols = np.zeros((pts.shape[0], 3), dtype=np.float64)
            cols[mask] = colors[mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            return pcd

        # Show each known label once: all its clusters together
        for lab_id, lab in enumerate(labels_in_order):
            cids = label_to_clusters.get(lab, [])
            if not cids:
                continue
            mask = np.isin(clusters, np.array(cids, dtype=np.int32))

            colors = np.zeros((pts.shape[0], 3), dtype=np.float64)
            colors[mask] = known_rgb01[lab_id]

            pcd = make_pcd(mask, colors)
            title = f"Label: {lab}  (clusters={cids})"
            print(f"\n[VIS] {title}")
            o3d.visualization.draw_geometries([pcd], window_name=title)

        # Show unknowns together: all unknown clusters shown in one view.
        if unknown_keys:
            # Color unknown points by unknown group key (unknown_0, unknown_1...) to separate them.
            colors = np.zeros((pts.shape[0], 3), dtype=np.float64)
            mask_any = np.zeros((pts.shape[0],), dtype=bool)

            for i, uk in enumerate(unknown_keys):
                cids = label_to_clusters.get(uk, [])
                if not cids:
                    continue
                m = np.isin(clusters, np.array(cids, dtype=np.int32))
                colors[m] = unknown_colors[i % unknown_colors.shape[0]]
                mask_any |= m

            pcd = make_pcd(mask_any, colors)
            title = f"Unknown clusters together (groups={len(unknown_keys)}, clusters={unknown_clusters_final})"
            print(f"\n[VIS] {title}")
            o3d.visualization.draw_geometries([pcd], window_name=title)
        else:
            print("\n[VIS] No unknown clusters left (all clusters assigned to known labels).")


    # ------------------------------------------------------------------
    # Save final label assignment (PLY + NPY + JSON)
    # ------------------------------------------------------------------

    SAVE_DIR = os.path.join(OVERLAY_DIR, "label_assignment_k20")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Build final label list (known + unknowns)
    unknown_keys = sorted([k for k in label_to_clusters.keys() if k.startswith("unknown_")])
    labels_extended = list(labels_in_order) + unknown_keys

    label_name_to_id = {name: i for i, name in enumerate(labels_extended)}
    label_id_to_name = {i: name for i, name in enumerate(labels_extended)}

    # Per-cluster -> label id
    cluster_to_label_id = {
        int(cid): label_name_to_id[name]
        for cid, name in cluster_to_assigned.items()
    }

    # Per-point label id
    assigned_label_ids = np.empty((clusters.shape[0],), dtype=np.int32)
    for cid in np.unique(clusters):
        cid = int(cid)
        mask = (clusters == cid)
        assigned_label_ids[mask] = cluster_to_label_id[cid]

    np.save(os.path.join(SAVE_DIR, "assigned_label_ids.npy"), assigned_label_ids)

    # ---------- Semantic JSON ----------
    label_counts = {}
    for i, name in enumerate(labels_extended):
        label_counts[name] = int((assigned_label_ids == i).sum())

    semantic_json = {
        "labels_in_order_known": labels_in_order,
        "labels_in_order_extended": labels_extended,
        "label_name_to_id": label_name_to_id,
        "label_id_to_name": label_id_to_name,
        "cluster_to_label_name": {str(k): v for k, v in cluster_to_assigned.items()},
        "label_to_clusters": {k: v for k, v in label_to_clusters.items()},
        "label_point_counts": label_counts,
        "n_points_total": int(len(assigned_label_ids)),
    }

    with open(os.path.join(SAVE_DIR, "labels_semantic.json"), "w") as f:
        json.dump(semantic_json, f, indent=2)

    # ---------- Build colors for all labels ----------
    all_colors = np.zeros((len(labels_extended), 3), dtype=np.float64)

    # known labels
    for i in range(len(labels_in_order)):
        all_colors[i] = known_rgb01[i]

    # unknown labels
    if unknown_keys:
        unknown_cols = distinct_colors_rgb01(len(unknown_keys))
        for j, uk in enumerate(unknown_keys):
            all_colors[label_name_to_id[uk]] = unknown_cols[j]

    # ---------- Save full colored PLY ----------
    full_colors = all_colors[assigned_label_ids]

    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(pts)
    pcd_all.colors = o3d.utility.Vector3dVector(full_colors)

    full_ply_path = os.path.join(SAVE_DIR, "assignment_colored.ply")
    o3d.io.write_point_cloud(full_ply_path, pcd_all)

    # ---------- Save per-label PLY ----------
    for name, lid in label_name_to_id.items():
        mask = (assigned_label_ids == lid)
        if not np.any(mask):
            continue

        pcd_lab = o3d.geometry.PointCloud()
        pcd_lab.points = o3d.utility.Vector3dVector(pts[mask])

        col = all_colors[lid]
        pcd_lab.colors = o3d.utility.Vector3dVector(
            np.tile(col[None, :], (int(mask.sum()), 1))
        )

        safe_name = name.replace("/", "_")
        out_path = os.path.join(SAVE_DIR, f"label_{safe_name}.ply")
        o3d.io.write_point_cloud(out_path, pcd_lab)

    print("\n[SAVE] Label assignment written to:")
    print(" ", SAVE_DIR)
    print("  - assigned_label_ids.npy")
    print("  - labels_semantic.json")
    print("  - assignment_colored.ply")
    print("  - label_<name>.ply (per label)")


if __name__ == "__main__":
    main()
