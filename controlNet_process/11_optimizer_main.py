#!/usr/bin/env python3
import os

from constraints_optimization.no_overlapping import apply_no_overlapping_shrink_only
from constraints_optimization.vis_bbx_before_after import (
    run_before_after_vis_per_label,
    run_global_vis_all_boxes,
)


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
    print("\n[OPT] === Shrink-only no-overlap + maximize size (tolerant, nonlinear) ===")
    print("[OPT] primitives:", PRIMITIVES_JSON)
    print("[OPT] ply       :", PLY_PATH)
    print("[OPT] cluster_ids:", CLUSTER_IDS_NPY)
    print("[OPT] out_dir   :", OUT_DIR)

    if not os.path.exists(PRIMITIVES_JSON):
        raise FileNotFoundError(f"Missing primitives json: {PRIMITIVES_JSON}")
    if not os.path.exists(PLY_PATH):
        raise FileNotFoundError(f"Missing ply: {PLY_PATH}")
    if not os.path.exists(CLUSTER_IDS_NPY):
        raise FileNotFoundError(f"Missing cluster ids npy: {CLUSTER_IDS_NPY}")

    outputs = apply_no_overlapping_shrink_only(
        primitives_json_path=PRIMITIVES_JSON,
        out_dir=OUT_DIR,
        steps=1200,
        lr=1e-2,
        overlap_tol_ratio=0.02,
        overlap_scale_ratio=0.01,
        r_min=0.70,
        w_overlap=1.0,
        w_cut=50.0,
        w_size=2.0,
        w_floor=10.0,
        extent_floor=1e-3,
        min_points=10,
        device="cuda",
        verbose_every=50,
    )

    print("\n[OPT] Done.")
    print("[OPT] optimized_primitives:", outputs["optimized_primitives_json"])
    print("[OPT] report              :", outputs["report_json"])

    print("\n[VIS] Per-label before/after views...")
    run_before_after_vis_per_label(
        before_primitives_json=PRIMITIVES_JSON,
        after_primitives_json=outputs["optimized_primitives_json"],
        ply_path=PLY_PATH,
        cluster_ids_path=CLUSTER_IDS_NPY,
    )

    print("\n[VIS] Global view: all optimized boxes together...")
    run_global_vis_all_boxes(
        primitives_json=outputs["optimized_primitives_json"],
        ply_path=PLY_PATH,
        show_points=True,
    )


if __name__ == "__main__":
    main()