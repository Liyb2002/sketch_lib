#!/usr/bin/env python3
import os
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# This code maps each cluster to a semantic label.
# clusters can be split in this code

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

SCENE_DIR = os.path.join(ROOT_DIR, "sketch", "3d_reconstruction")

# Your merged label output folder from the previous script
PARTFIELD_OVERLAY_DIR = os.path.join(ROOT_DIR, "sketch", "partfield_overlay")

# Inputs
FUSED_PLY_PATH = os.path.join(SCENE_DIR, "fused_model.ply")
CLUSTERS_PATH  = os.path.join(SCENE_DIR, "clustering_k20.npy")

MERGED_LABELED_PLY = os.path.join(PARTFIELD_OVERLAY_DIR, "merged_labeled.ply")
LABEL_COLOR_MAP_JSON = os.path.join(PARTFIELD_OVERLAY_DIR, "label_color_map.json")

# Outputs (requested)
OUTPUT_DIR = os.path.join(ROOT_DIR, "sketch", "clusters")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_PLY_PATH  = os.path.join(OUTPUT_DIR, "labeled_clusters.ply")
FINAL_JSON_PATH = os.path.join(OUTPUT_DIR, "cluster_to_label.json")

# NEW: per-point cluster ids aligned to labeled_clusters.ply
FINAL_NPY_PATH  = os.path.join(OUTPUT_DIR, "final_cluster_ids.npy")

# Thresholds (requested)
MERGE_THRESHOLD  = 0.90
IGNORE_THRESHOLD = 0.10

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def generate_palette(unique_labels):
    """Stable palette for labels."""
    palette = {}
    cmap = plt.get_cmap("tab20")
    sorted_labels = sorted(list(unique_labels))
    for i, label in enumerate(sorted_labels):
        if label == "unknown":
            palette[label] = [0.2, 0.2, 0.2]
        else:
            palette[label] = list(cmap(i % 20)[:3])  # RGB in [0,1]
    return palette

def parse_label_color_map(path):
    """
    Expects label_color_map.json created by your voting script, containing:
      - labels_in_order
      - label_to_color_rgb (0-255 ints in RGB order)
      - color_bgr_to_label (string key "b,g,r")
    We'll mainly use color_bgr_to_label for exact matching.
    """
    with open(path, "r") as f:
        m = json.load(f)

    color_bgr_to_label = m.get("color_bgr_to_label", {})
    labels_in_order = m.get("labels_in_order", [])

    # Build float RGB [0,1] reference for each label (for fallback nearest match)
    label_to_rgb01 = {}
    label_to_color_rgb = m.get("label_to_color_rgb", {})
    for lab, rgb in label_to_color_rgb.items():
        # rgb stored as [r,g,b] ints
        r, g, b = rgb
        label_to_rgb01[lab] = np.array([r, g, b], dtype=np.float32) / 255.0

    return color_bgr_to_label, labels_in_order, label_to_rgb01

def label_from_color_rgb01(rgb01, color_bgr_to_label, label_to_rgb01, max_dist=0.05):
    """
    Convert a point color (rgb in [0,1]) into a label.
    Prefer exact match through BGR string if possible, otherwise nearest match.
    """
    # Treat near-black as unknown (matches your earlier pipeline behavior)
    if float(rgb01.sum()) < 0.05:
        return "unknown"

    # Try exact match: convert to uint8 and lookup by BGR string
    rgb255 = np.clip(np.round(rgb01 * 255.0), 0, 255).astype(np.uint8)
    r, g, b = int(rgb255[0]), int(rgb255[1]), int(rgb255[2])
    key = f"{b},{g},{r}"  # BGR string key
    if key in color_bgr_to_label:
        return color_bgr_to_label[key]

    # Fallback: nearest label color in RGB01
    best = "unknown"
    best_d = max_dist
    for lab, ref in label_to_rgb01.items():
        d = float(np.linalg.norm(rgb01 - ref))
        if d < best_d:
            best_d = d
            best = lab
    return best

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

    # -------------------------------------------------------------------------
    # 1) Load merged labeled point cloud (already voted across views)
    # -------------------------------------------------------------------------
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

    # Convert per-point color -> label
    raw_point_labels = np.array(["unknown"] * num_points, dtype=object)
    for i in range(num_points):
        raw_point_labels[i] = label_from_color_rgb01(
            merged_colors[i].astype(np.float32),
            color_bgr_to_label=color_bgr_to_label,
            label_to_rgb01=label_to_rgb01,
            max_dist=0.05,
        )

    global_semantic_registry = set(labels_in_order) if labels_in_order else set(raw_point_labels.tolist())
    global_semantic_registry.add("unknown")

    # -------------------------------------------------------------------------
    # 2) Apply your 90/10 cluster regrouping rules
    # -------------------------------------------------------------------------
    print("[INFO] Applying cluster regrouping rules (>=90% merge, 10-90 split, <=10 remove)...")

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

        # (1) >= 90% => whole cluster
        if dom_ratio >= MERGE_THRESHOLD:
            refined_labels[idxs] = dom_label
            continue

        # (2)/(3) Otherwise:
        # - 10% < ratio < 90% => keep as that label (will be split to new cluster)
        # - ratio <= 10% => remove those points => unknown
        for lab, ratio in ratios.items():
            if ratio <= IGNORE_THRESHOLD:
                kill = idxs[refined_labels[idxs] == lab]
                refined_labels[kill] = "unknown"

    # -------------------------------------------------------------------------
    # 3) Create final clusters: one new cluster per (orig_cluster_id, label) excluding unknown
    # -------------------------------------------------------------------------
    print("[INFO] Creating final cluster ids + registry...")

    palette = generate_palette(global_semantic_registry)

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
                "color_rgb": [float(x) for x in palette.get(lab, [0.2, 0.2, 0.2])]
            }
            new_id += 1

    # -------------------------------------------------------------------------
    # 4) Export labeled shape + json mapping + NEW npy cluster ids
    # -------------------------------------------------------------------------
    with open(FINAL_JSON_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[SAVE] {FINAL_JSON_PATH}")

    # NEW: save per-point final cluster ids aligned to labeled_clusters.ply
    np.save(FINAL_NPY_PATH, final_cluster_ids)
    print(f"[SAVE] {FINAL_NPY_PATH} (shape={final_cluster_ids.shape}, dtype={final_cluster_ids.dtype})")

    unknown_color = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    out_colors = np.zeros((num_points, 3), dtype=np.float32)
    for i in range(num_points):
        cid = final_cluster_ids[i]
        if cid >= 0:
            lab = registry[cid]["label"]
            out_colors[i] = np.array(palette.get(lab, [0.2, 0.2, 0.2]), dtype=np.float32)
        else:
            out_colors[i] = unknown_color

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    pcd_out.colors = o3d.utility.Vector3dVector(out_colors.astype(np.float64))
    o3d.io.write_point_cloud(FINAL_PLY_PATH, pcd_out)
    print(f"[SAVE] {FINAL_PLY_PATH}")

    print(f"[OK] Final labeled clusters: {len(registry)}")
    print("[OK] Done.")

if __name__ == "__main__":
    main()
