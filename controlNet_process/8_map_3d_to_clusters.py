#!/usr/bin/env python3
import os
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# This code maps each cluster to a semantic label.
# clusters can be split in this code
# NEW: keep unlabeled regions as explicit clusters: unknown_0, unknown_1, ...
# NEW: merge neighboring unknown clusters at CLUSTER level using AABB distance.
#
# IMPORTANT CHANGE (per your note):
# - BLACK color means "no label at all".
# - If original cluster A and B are entirely black, they MUST become different unknown labels
#   (unknown_0, unknown_1, ...) and should NOT be merged.
# - Different labels => different colors (unknown_* get distinct colors).
#
# MINIMAL FIX (per your last request):
# - Do NOT leave any point with final_cluster_id = -1.
# - Any leftover -1 points get assigned to ONE explicit unknown_* cluster (single bucket),
#   so we do NOT explode the number of clusters.

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

SCENE_DIR = os.path.join(ROOT_DIR, "sketch", "3d_reconstruction")
PARTFIELD_OVERLAY_DIR = os.path.join(ROOT_DIR, "sketch", "partfield_overlay")

# Inputs
FUSED_PLY_PATH = os.path.join(SCENE_DIR, "fused_model.ply")
CLUSTERS_PATH  = os.path.join(SCENE_DIR, "clustering_k20.npy")

MERGED_LABELED_PLY = os.path.join(PARTFIELD_OVERLAY_DIR, "merged_labeled.ply")
LABEL_COLOR_MAP_JSON = os.path.join(PARTFIELD_OVERLAY_DIR, "label_color_map.json")

# Outputs
OUTPUT_DIR = os.path.join(ROOT_DIR, "sketch", "clusters")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_PLY_PATH  = os.path.join(OUTPUT_DIR, "labeled_clusters.ply")
FINAL_JSON_PATH = os.path.join(OUTPUT_DIR, "cluster_to_label.json")
FINAL_NPY_PATH  = os.path.join(OUTPUT_DIR, "final_cluster_ids.npy")

# Thresholds
MERGE_THRESHOLD  = 0.90
IGNORE_THRESHOLD = 0.10

# Unknown cluster merging threshold (AABB gap)
UNKNOWN_MERGE_GAP_AUTO = True          # recommended
UNKNOWN_MERGE_GAP_FRAC = 0.01          # gap = bbox_diagonal * frac
UNKNOWN_MERGE_GAP_ABS  = 0.02          # used if AUTO=False (units of your point cloud)

# Color threshold for "BLACK means no label at all"
BLACK_SUM_THRESH = 0.05

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def generate_palette(unique_labels):
    """
    Stable palette for labels, including unknown_*.
    Different labels => different colors.
    """
    palette = {}
    cmap = plt.get_cmap("tab20")
    sorted_labels = sorted([str(x) for x in unique_labels])

    for i, label in enumerate(sorted_labels):
        palette[label] = list(cmap(i % 20)[:3])  # RGB in [0,1]

    return palette

def parse_label_color_map(path):
    """
    Expects label_color_map.json containing:
      - labels_in_order
      - label_to_color_rgb (0-255 ints in RGB order)
      - color_bgr_to_label (string key "b,g,r")
    """
    with open(path, "r") as f:
        m = json.load(f)

    color_bgr_to_label = m.get("color_bgr_to_label", {})
    labels_in_order = m.get("labels_in_order", [])

    label_to_rgb01 = {}
    label_to_color_rgb = m.get("label_to_color_rgb", {})
    for lab, rgb in label_to_color_rgb.items():
        r, g, b = rgb
        label_to_rgb01[lab] = np.array([r, g, b], dtype=np.float32) / 255.0

    return color_bgr_to_label, labels_in_order, label_to_rgb01

def aabb_of_points(pts):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return mn, mx

def aabb_gap_distance(a_min, a_max, b_min, b_max):
    """Euclidean distance between two AABBs (0 if they overlap/touch)."""
    dx = max(0.0, max(b_min[0] - a_max[0], a_min[0] - b_max[0]))
    dy = max(0.0, max(b_min[1] - a_max[1], a_min[1] - b_max[1]))
    dz = max(0.0, max(b_min[2] - a_max[2], a_min[2] - b_max[2]))
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))

def union_find_init(n):
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    return find, union, parent

def colors_rgb01_to_labels_fast(merged_colors_rgb01, color_bgr_to_label, label_to_rgb01, max_dist=0.05, black_sum_thresh=0.05):
    """
    Fast color->label:
    - Compute black mask: sum(rgb) < black_sum_thresh => NO LABEL AT ALL.
    - For non-black points: try exact map via BGR dict; else nearest label color.
    Returns:
      labels: (N,) object array (black points are set to "unknown")
      is_black: (N,) bool array
    """
    cols = np.asarray(merged_colors_rgb01, dtype=np.float32)
    num_points = cols.shape[0]
    if num_points == 0:
        return np.array([], dtype=object), np.array([], dtype=bool)

    is_black = (cols.sum(axis=1) < float(black_sum_thresh))

    rgb255 = np.clip(np.rint(cols * 255.0), 0, 255).astype(np.uint8)
    r = rgb255[:, 0].astype(np.uint32)
    g = rgb255[:, 1].astype(np.uint32)
    b = rgb255[:, 2].astype(np.uint32)

    packed = (r << 16) | (g << 8) | b

    uniq_packed, inv = np.unique(packed, return_inverse=True)
    uniq_labels = np.full((uniq_packed.shape[0],), "unknown", dtype=object)

    ur = ((uniq_packed >> 16) & 255).astype(np.uint8)
    ug = ((uniq_packed >> 8) & 255).astype(np.uint8)
    ub = (uniq_packed & 255).astype(np.uint8)

    for k in range(uniq_packed.shape[0]):
        key = f"{int(ub[k])},{int(ug[k])},{int(ur[k])}"
        lab = color_bgr_to_label.get(key, None)
        if lab is not None:
            uniq_labels[k] = lab

    unknown_mask = (uniq_labels == "unknown")
    if np.any(unknown_mask) and len(label_to_rgb01) > 0:
        ref_labels = list(label_to_rgb01.keys())
        ref_colors = np.stack([label_to_rgb01[l] for l in ref_labels], axis=0).astype(np.float32)

        uniq_rgb01 = np.stack([ur, ug, ub], axis=1).astype(np.float32) / 255.0
        uq = uniq_rgb01[unknown_mask]

        dists = np.linalg.norm(uq[:, None, :] - ref_colors[None, :, :], axis=2)
        best_idx = np.argmin(dists, axis=1)
        best_dist = dists[np.arange(dists.shape[0]), best_idx]

        accepted = best_dist < float(max_dist)
        fill = np.array(["unknown"] * uq.shape[0], dtype=object)
        if np.any(accepted):
            fill[accepted] = np.array([ref_labels[i] for i in best_idx[accepted]], dtype=object)

        uniq_labels[unknown_mask] = fill

    labels = uniq_labels[inv]
    labels[is_black] = "unknown"
    return labels, is_black

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print(f"[INFO] Loading geometry: {FUSED_PLY_PATH}")
    if not os.path.exists(FUSED_PLY_PATH):
        raise RuntimeError(f"Missing fused model: {FUSED_PLY_PATH}")

    pcd_orig = o3d.io.read_point_cloud(FUSED_PLY_PATH)
    points = np.asarray(pcd_orig.points)
    num_points = points.shape[0]
    print(f"[INFO] Points: {num_points}")

    print(f"[INFO] Loading original clusters: {CLUSTERS_PATH}")
    if not os.path.exists(CLUSTERS_PATH):
        raise RuntimeError(f"Missing clusters file: {CLUSTERS_PATH}")
    orig_cluster_ids = np.load(CLUSTERS_PATH).reshape(-1)
    if orig_cluster_ids.shape[0] != num_points:
        raise RuntimeError(
            f"Cluster id length {orig_cluster_ids.shape[0]} != fused points {num_points}. "
            "Your clustering must be generated from fused_model.ply in the same point order."
        )

    print(f"[INFO] Loading merged labeled PLY: {MERGED_LABELED_PLY}")
    if not os.path.exists(MERGED_LABELED_PLY):
        raise RuntimeError(f"Missing merged labeled ply: {MERGED_LABELED_PLY}")

    pcd_merged = o3d.io.read_point_cloud(MERGED_LABELED_PLY)
    merged_colors = np.asarray(pcd_merged.colors)
    if merged_colors.shape[0] != num_points:
        raise RuntimeError(
            f"merged_labeled.ply points {merged_colors.shape[0]} != fused_model points {num_points}. "
            "These must match point-for-point."
        )

    print(f"[INFO] Loading label-color map: {LABEL_COLOR_MAP_JSON}")
    if not os.path.exists(LABEL_COLOR_MAP_JSON):
        raise RuntimeError(f"Missing label_color_map.json: {LABEL_COLOR_MAP_JSON}")

    color_bgr_to_label, labels_in_order, label_to_rgb01 = parse_label_color_map(LABEL_COLOR_MAP_JSON)

    raw_point_labels, is_black = colors_rgb01_to_labels_fast(
        merged_colors,
        color_bgr_to_label=color_bgr_to_label,
        label_to_rgb01=label_to_rgb01,
        max_dist=0.05,
        black_sum_thresh=BLACK_SUM_THRESH,
    )

    global_semantic_registry = set(labels_in_order) if labels_in_order else set(raw_point_labels.tolist())
    global_semantic_registry.add("unknown")

    # -------------------------------------------------------------------------
    # 2) Apply your 90/10 cluster regrouping rules
    # -------------------------------------------------------------------------
    print("[INFO] Applying cluster regrouping rules (>=90% merge, <=10 remove -> unknown; 10-90 kept as-is)...")

    refined_labels = raw_point_labels.copy()
    unique_orig_clusters = np.unique(orig_cluster_ids)

    for cid in unique_orig_clusters:
        if cid < 0:
            continue

        idxs = np.where(orig_cluster_ids == cid)[0]
        total = idxs.shape[0]
        if total == 0:
            continue

        lbls = refined_labels[idxs]
        non_unknown = lbls[lbls != "unknown"]
        if non_unknown.size == 0:
            continue

        uniq, cnts = np.unique(non_unknown, return_counts=True)
        ratios = {lab: (c / total) for lab, c in zip(uniq, cnts)}

        dom_label = max(ratios, key=ratios.get)
        dom_ratio = ratios[dom_label]

        if dom_ratio >= MERGE_THRESHOLD:
            refined_labels[idxs] = dom_label
            continue

        for lab, ratio in ratios.items():
            if ratio <= IGNORE_THRESHOLD:
                kill = idxs[refined_labels[idxs] == lab]
                refined_labels[kill] = "unknown"

    # -------------------------------------------------------------------------
    # 3) Create labeled clusters: one new cluster per (orig_cluster_id, label) excluding unknown
    # -------------------------------------------------------------------------
    print("[INFO] Creating labeled clusters + registry (excluding unknown for now)...")

    final_cluster_ids = np.full((num_points,), -1, dtype=np.int32)
    registry = {}
    new_id = 0

    for cid in unique_orig_clusters:
        if cid < 0:
            continue
        idxs = np.where(orig_cluster_ids == cid)[0]
        if idxs.size == 0:
            continue

        labs = np.unique(refined_labels[idxs])
        for lab in labs:
            if lab == "unknown":
                continue
            sub = idxs[refined_labels[idxs] == lab]
            if sub.size == 0:
                continue

            final_cluster_ids[sub] = new_id
            registry[new_id] = {
                "label": str(lab),
                "original_cluster_id": int(cid),
                "point_count": int(sub.size),
                "color_rgb": None
            }
            new_id += 1

    # -------------------------------------------------------------------------
    # 4) Unknown entities at CLUSTER level, then merge neighboring unknown clusters
    # -------------------------------------------------------------------------
    print("[INFO] Creating unknown entities at cluster level...")

    unknown_entities = []
    for cid in unique_orig_clusters:
        if cid < 0:
            continue
        idxs = np.where(orig_cluster_ids == cid)[0]
        if idxs.size == 0:
            continue

        unk = idxs[refined_labels[idxs] == "unknown"]
        if unk.size == 0:
            continue

        pure_black_cluster = bool(np.all(is_black[idxs]))

        pts = points[unk]
        mn, mx = aabb_of_points(pts)
        unknown_entities.append({
            "orig_cid": int(cid),
            "idxs": unk,
            "aabb_min": mn,
            "aabb_max": mx,
            "pure_black_cluster": pure_black_cluster,
        })

    print(f"[INFO] Unknown entities (pre-merge): {len(unknown_entities)}")

    if UNKNOWN_MERGE_GAP_AUTO:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        merge_gap = max(1e-8, diag * float(UNKNOWN_MERGE_GAP_FRAC))
    else:
        merge_gap = float(UNKNOWN_MERGE_GAP_ABS)

    print(f"[INFO] Unknown merge gap = {merge_gap:.6f} (auto={UNKNOWN_MERGE_GAP_AUTO})")

    if len(unknown_entities) > 0:
        find, union, _parent = union_find_init(len(unknown_entities))

        for i in range(len(unknown_entities)):
            ai = unknown_entities[i]
            for j in range(i + 1, len(unknown_entities)):
                aj = unknown_entities[j]

                if ai["pure_black_cluster"] and aj["pure_black_cluster"]:
                    continue

                d = aabb_gap_distance(ai["aabb_min"], ai["aabb_max"], aj["aabb_min"], aj["aabb_max"])
                if d <= merge_gap:
                    union(i, j)

        roots = np.array([find(i) for i in range(len(unknown_entities))], dtype=np.int32)
        uniq_roots, inv = np.unique(roots, return_inverse=True)

        merged_groups = []
        for gid in range(len(uniq_roots)):
            members = np.where(inv == gid)[0]
            merged_groups.append(members)

        print(f"[INFO] Unknown groups (post-merge): {len(merged_groups)}")

        unknown_labels = [f"unknown_{i}" for i in range(len(merged_groups))]
    else:
        merged_groups = []
        unknown_labels = []

    # -------------------------------------------------------------------------
    # 5) Final palette (includes unknown_* labels)
    # -------------------------------------------------------------------------
    palette_labels = set(global_semantic_registry)
    palette_labels.discard("unknown")
    palette_labels.update(unknown_labels)
    for v in registry.values():
        palette_labels.add(v["label"])

    palette = generate_palette(palette_labels)

    for cid_out, info in registry.items():
        lab = info["label"]
        info["color_rgb"] = [float(x) for x in palette.get(lab, [0.2, 0.2, 0.2])]

    # -------------------------------------------------------------------------
    # 6) Create unknown_* clusters in registry + assign ids
    # -------------------------------------------------------------------------
    for ui, members in enumerate(merged_groups):
        lab = f"unknown_{ui}"
        idxs_list = [unknown_entities[m]["idxs"] for m in members]
        sub = np.concatenate(idxs_list, axis=0) if len(idxs_list) > 1 else idxs_list[0]

        cid_new = new_id
        final_cluster_ids[sub] = cid_new
        registry[cid_new] = {
            "label": lab,
            "original_cluster_id": -1,
            "point_count": int(sub.size),
            "color_rgb": [float(x) for x in palette.get(lab, [0.2, 0.2, 0.2])]
        }
        new_id += 1

    # -------------------------------------------------------------------------
    # 7) MINIMAL FIX (CHANGED): if any points still -1, assign them ALL to ONE extra unknown_*
    #     (instead of creating many unknowns by orig_cluster_id)
    # -------------------------------------------------------------------------
    leftover = np.where(final_cluster_ids < 0)[0]
    if leftover.size > 0:
        print(f"[INFO] Assigning ONE explicit unknown_* id for {leftover.size} leftover points...")

        existing_unknown = sum(1 for v in registry.values() if str(v.get("label", "")).startswith("unknown_"))
        lab = f"unknown_{existing_unknown}"

        # ensure palette has this label
        if lab not in palette:
            palette2 = set(palette.keys())
            palette2.add(lab)
            palette = generate_palette(palette2)

        cid_new = new_id
        final_cluster_ids[leftover] = cid_new
        registry[cid_new] = {
            "label": lab,
            "original_cluster_id": -1,
            "point_count": int(leftover.size),
            "color_rgb": [float(x) for x in palette.get(lab, [0.2, 0.2, 0.2])]
        }
        new_id += 1

        leftover2 = np.where(final_cluster_ids < 0)[0]
        if leftover2.size > 0:
            raise RuntimeError(f"[FATAL] Still {leftover2.size} points have cluster_id=-1 after unknown_* assignment.")

    # -------------------------------------------------------------------------
    # 8) Export json + npy + ply
    # -------------------------------------------------------------------------
    with open(FINAL_JSON_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[SAVE] {FINAL_JSON_PATH}")

    np.save(FINAL_NPY_PATH, final_cluster_ids)
    print(f"[SAVE] {FINAL_NPY_PATH} (shape={final_cluster_ids.shape}, dtype={final_cluster_ids.dtype})")

    max_cid = int(final_cluster_ids.max())
    if max_cid < 0:
        raise RuntimeError("No clusters were assigned (final_cluster_ids.max() < 0).")

    color_table = np.zeros((max_cid + 1, 3), dtype=np.float32)
    for cid_out, info in registry.items():
        cid_out = int(cid_out)
        if 0 <= cid_out <= max_cid:
            lab = info["label"]
            color_table[cid_out] = np.array(palette.get(lab, [0.2, 0.2, 0.2]), dtype=np.float32)

    out_colors = color_table[final_cluster_ids]

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    pcd_out.colors = o3d.utility.Vector3dVector(out_colors.astype(np.float64))
    o3d.io.write_point_cloud(FINAL_PLY_PATH, pcd_out)
    print(f"[SAVE] {FINAL_PLY_PATH}")

    num_unknown_clusters = sum(1 for v in registry.values() if str(v.get("label", "")).startswith("unknown_"))
    print(f"[OK] Final clusters: {len(registry)} (unknown_* = {num_unknown_clusters})")
    print("[OK] Done.")

if __name__ == "__main__":
    main()
