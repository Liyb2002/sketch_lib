#!/usr/bin/env python3
import os

from constraints_extraction.heat_map import build_label_heatmaps

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")

CLUSTERS_DIR = os.path.join(SKETCH_ROOT, "clusters")
DSL_DIR      = os.path.join(SKETCH_ROOT, "dsl_optimize")

MERGED_PRIMITIVES = os.path.join(DSL_DIR, "merged_pca_primitives.json")
BASE_PRIMITIVES   = os.path.join(DSL_DIR, "pca_primitives.json")
PRIMITIVES_JSON   = MERGED_PRIMITIVES if os.path.exists(MERGED_PRIMITIVES) else BASE_PRIMITIVES

MERGED_PLY = os.path.join(DSL_DIR, "merged_labeled_clusters.ply")
BASE_PLY   = os.path.join(CLUSTERS_DIR, "labeled_clusters.ply")
PLY_PATH   = MERGED_PLY if os.path.exists(MERGED_PLY) else BASE_PLY

# cluster ids aligned to chosen ply
if os.path.basename(PLY_PATH) == "merged_labeled_clusters.ply":
    CLUSTER_IDS_NPY = os.path.join(DSL_DIR, "merged_cluster_ids.npy")
else:
    CLUSTER_IDS_NPY = os.path.join(CLUSTERS_DIR, "final_cluster_ids.npy")

ITER_ID = 0
OUT_DIR = os.path.join(DSL_DIR, "optimize_iteration", f"iter_{ITER_ID:03d}")


def main():
    print("\n[OPT] === Step 1: voxel heat maps per label (label fraction per voxel) ===")
    print("[OPT] primitives :", PRIMITIVES_JSON)
    print("[OPT] ply        :", PLY_PATH)
    print("[OPT] cluster_ids:", CLUSTER_IDS_NPY)
    print("[OPT] out_dir    :", OUT_DIR)

    if not os.path.exists(PRIMITIVES_JSON):
        raise FileNotFoundError(f"Missing primitives json: {PRIMITIVES_JSON}")
    if not os.path.exists(PLY_PATH):
        raise FileNotFoundError(f"Missing ply: {PLY_PATH}")
    if not os.path.exists(CLUSTER_IDS_NPY):
        raise FileNotFoundError(f"Missing cluster ids npy: {CLUSTER_IDS_NPY}")

    heat_dir = os.path.join(OUT_DIR, "heat_map")
    build_label_heatmaps(
        primitives_json_path=PRIMITIVES_JSON,
        ply_path=PLY_PATH,
        cluster_ids_path=CLUSTER_IDS_NPY,
        out_dir=heat_dir,
        voxel_size=None,              # auto
        min_points_per_label=200,
        show_windows=True,            # pops Open3D windows per label
        max_labels_to_show=12,
        show_combined=True,
    )


if __name__ == "__main__":
    main()