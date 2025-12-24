#!/usr/bin/env python3
import os
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIG (hard-coded)
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
LABEL_ID_JSON   = os.path.join(OUTPUT_DIR, "label_to_id.json")

# 90/10 rules
MERGE_THRESHOLD  = 0.90
IGNORE_THRESHOLD = 0.10

# Color threshold: black means no label from overlays
BLACK_SUM_THRESH = 0.05

# Label ids start here
LABEL_ID_START = 21


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def generate_palette(unique_names):
    cmap = plt.get_cmap("tab20")
    names = sorted([str(x) for x in unique_names])
    pal = {}
    for i, n in enumerate(names):
        pal[n] = list(cmap(i % 20)[:3])
    return pal

def parse_label_color_map(path):
    with open(path, "r") as f:
        m = json.load(f)

    color_bgr_to_label = m.get("color_bgr_to_label", {})
    labels_in_order = m.get("labels_in_order", [])

    label_to_rgb01 = {}
    label_to_color_rgb = m.get("label_to_color_rgb", {})
    for lab, rgb in label_to_color_rgb.items():
        r, g, b = rgb
        label_to_rgb01[str(lab)] = np.array([r, g, b], dtype=np.float32) / 255.0

    return color_bgr_to_label, labels_in_order, label_to_rgb01

def colors_rgb01_to_labels_fast(merged_colors_rgb01, color_bgr_to_label, label_to_rgb01,
                                max_dist=0.05, black_sum_thresh=0.05):
    cols = np.asarray(merged_colors_rgb01, dtype=np.float32)
    N = cols.shape[0]
    if N == 0:
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
            uniq_labels[k] = str(lab)

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
    print(f"[INFO] Loading fused model: {FUSED_PLY_PATH}")
    if not os.path.exists(FUSED_PLY_PATH):
        raise RuntimeError(f"Missing: {FUSED_PLY_PATH}")
    pcd = o3d.io.read_point_cloud(FUSED_PLY_PATH)
    points = np.asarray(pcd.points)
    N = points.shape[0]
    print(f"[INFO] Points: {N}")

    print(f"[INFO] Loading initial clustering: {CLUSTERS_PATH}")
    if not os.path.exists(CLUSTERS_PATH):
        raise RuntimeError(f"Missing: {CLUSTERS_PATH}")
    orig_cluster_ids = np.load(CLUSTERS_PATH).reshape(-1).astype(np.int64)
    if orig_cluster_ids.shape[0] != N:
        raise RuntimeError(f"clustering_k20.npy len {orig_cluster_ids.shape[0]} != points {N}")
    if np.any(orig_cluster_ids < 0):
        bad = int(np.sum(orig_cluster_ids < 0))
        raise RuntimeError(f"[FATAL] Initial clustering has {bad} ids < 0. Fix upstream first.")

    unique_orig = np.unique(orig_cluster_ids)
    print(f"[OK] Initial clustering clean. orig clusters: {len(unique_orig)}")

    print(f"[INFO] Loading merged labeled PLY (overlays): {MERGED_LABELED_PLY}")
    if not os.path.exists(MERGED_LABELED_PLY):
        raise RuntimeError(f"Missing: {MERGED_LABELED_PLY}")
    pcd_overlay = o3d.io.read_point_cloud(MERGED_LABELED_PLY)
    overlay_colors = np.asarray(pcd_overlay.colors)
    if overlay_colors.shape[0] != N:
        raise RuntimeError(f"merged_labeled.ply points {overlay_colors.shape[0]} != fused points {N}")

    print(f"[INFO] Loading label_color_map.json: {LABEL_COLOR_MAP_JSON}")
    if not os.path.exists(LABEL_COLOR_MAP_JSON):
        raise RuntimeError(f"Missing: {LABEL_COLOR_MAP_JSON}")

    color_bgr_to_label, labels_in_order, label_to_rgb01 = parse_label_color_map(LABEL_COLOR_MAP_JSON)

    raw_point_labels, is_black = colors_rgb01_to_labels_fast(
        overlay_colors,
        color_bgr_to_label=color_bgr_to_label,
        label_to_rgb01=label_to_rgb01,
        max_dist=0.05,
        black_sum_thresh=BLACK_SUM_THRESH,
    )

    # -------------------------------------------------------------------------
    # 1) Refine point labels within each original cluster + record forced clusters
    # -------------------------------------------------------------------------
    print("[INFO] Applying 90/10 refinement inside each original cluster...")
    refined_labels = raw_point_labels.copy()

    # NEW: record cluster-level forced label if dominant >= 0.90
    cluster_force_label = {}  # cid(int) -> label(str)

    for cid in unique_orig:
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

        # If dominant label >= 0.90, force WHOLE cluster to that label
        if dom_ratio >= MERGE_THRESHOLD:
            refined_labels[idxs] = dom_label
            cluster_force_label[int(cid)] = str(dom_label)
            continue

        # Otherwise, apply <=10% kill rule
        for lab, ratio in ratios.items():
            if ratio <= IGNORE_THRESHOLD:
                kill = idxs[refined_labels[idxs] == lab]
                refined_labels[kill] = "unknown"

    print(f"[INFO] Forced whole-cluster labels: {len(cluster_force_label)}")

    # -------------------------------------------------------------------------
    # 2) Build label->id mapping starting from 21 (or > max_orig if needed)
    # -------------------------------------------------------------------------
    semantic_labels = sorted(set(map(str, refined_labels.tolist())) - {"unknown"})
    max_orig = int(orig_cluster_ids.max())
    start = int(LABEL_ID_START)
    if start <= max_orig:
        start = max_orig + 1
        print(f"[WARN] LABEL_ID_START={LABEL_ID_START} collides with orig ids up to {max_orig}. Using start={start} instead.")

    label_to_id = {lab: (start + i) for i, lab in enumerate(semantic_labels)}

    with open(LABEL_ID_JSON, "w") as f:
        json.dump(label_to_id, f, indent=2)
    print(f"[SAVE] {LABEL_ID_JSON}")

    # -------------------------------------------------------------------------
    # 3) Final ids:
    #    - default keep original cluster ids
    #    - if cluster forced: move ALL its points to label id
    #    - else: per-point labeled overwrite
    # -------------------------------------------------------------------------
    final_cluster_ids = orig_cluster_ids.astype(np.int32).copy()

    # Apply forced clusters first
    for cid, lab in cluster_force_label.items():
        if lab == "unknown":
            continue
        if lab not in label_to_id:
            # Should not happen, but be safe:
            label_to_id[lab] = int(max(label_to_id.values(), default=start - 1) + 1)
        lid = int(label_to_id[lab])
        mask = (orig_cluster_ids == cid)
        final_cluster_ids[mask] = lid

    # Then apply remaining per-point assignments (for non-forced clusters)
    for lab, lid in label_to_id.items():
        mask = (refined_labels == lab)
        final_cluster_ids[mask] = int(lid)

    if np.any(final_cluster_ids < 0):
        bad = int(np.sum(final_cluster_ids < 0))
        raise RuntimeError(f"[FATAL] Produced {bad} final ids < 0. This should never happen.")

    # -------------------------------------------------------------------------
    # 4) Build registry.json: include BOTH original cluster ids and label ids
    #    (Original ids that got fully forced will naturally have point_count=0)
    # -------------------------------------------------------------------------
    registry = {}

    for cid in unique_orig:
        cid = int(cid)
        count = int(np.sum(final_cluster_ids == cid))
        registry[str(cid)] = {
            "label": f"cluster_{cid}",
            "type": "orig_cluster",
            "point_count": count,
        }

    for lab, lid in label_to_id.items():
        lid = int(lid)
        count = int(np.sum(final_cluster_ids == lid))
        registry[str(lid)] = {
            "label": str(lab),
            "type": "semantic_label",
            "point_count": count,
        }

    with open(FINAL_JSON_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[SAVE] {FINAL_JSON_PATH}")

    np.save(FINAL_NPY_PATH, final_cluster_ids)
    print(f"[SAVE] {FINAL_NPY_PATH} (shape={final_cluster_ids.shape}, dtype={final_cluster_ids.dtype})")

    # -------------------------------------------------------------------------
    # 5) Color + export PLY
    # -------------------------------------------------------------------------
    palette_names = [v["label"] for v in registry.values()]
    palette = generate_palette(set(palette_names))

    max_id = int(final_cluster_ids.max())
    color_table = np.zeros((max_id + 1, 3), dtype=np.float32)

    for k, info in registry.items():
        cid = int(k)
        name = str(info["label"])
        if 0 <= cid <= max_id:
            color_table[cid] = np.array(palette.get(name, [0.2, 0.2, 0.2]), dtype=np.float32)

    out_colors = color_table[final_cluster_ids]

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    pcd_out.colors = o3d.utility.Vector3dVector(out_colors.astype(np.float64))
    o3d.io.write_point_cloud(FINAL_PLY_PATH, pcd_out)
    print(f"[SAVE] {FINAL_PLY_PATH}")

    print("[OK] Done.")
    print(f"[OK] orig ids: {len(unique_orig)} | semantic labels: {len(label_to_id)}")
    forced_nonzero = sum(1 for cid in unique_orig if int(np.sum(final_cluster_ids == int(cid))) == 0)
    print(f"[OK] orig clusters now empty (fully reassigned): {forced_nonzero}")

if __name__ == "__main__":
    main()
