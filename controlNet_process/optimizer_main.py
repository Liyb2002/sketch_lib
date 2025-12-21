#!/usr/bin/env python3
import os

from constraints_optimization.same_pair_relation import apply_same_pair_relation
from constraints_optimization.vis_bbx_before_after import run_before_after_vis_per_label


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")

CLUSTERS_DIR = os.path.join(SKETCH_ROOT, "clusters")
DSL_DIR      = os.path.join(SKETCH_ROOT, "dsl_optimize")

# ---------------------------------------------------------------------
# HARD-CODED INPUTS
# ---------------------------------------------------------------------

RELATIONS_JSON = os.path.join(DSL_DIR, "relations.json")

# Prefer merged primitives if they exist
MERGED_PRIMITIVES = os.path.join(DSL_DIR, "merged_pca_primitives.json")
BASE_PRIMITIVES   = os.path.join(DSL_DIR, "pca_primitives.json")
PRIMITIVES_JSON   = MERGED_PRIMITIVES if os.path.exists(MERGED_PRIMITIVES) else BASE_PRIMITIVES

# Prefer merged labeled PLY if it exists
MERGED_PLY = os.path.join(DSL_DIR, "merged_labeled_clusters.ply")
BASE_PLY   = os.path.join(CLUSTERS_DIR, "labeled_clusters.ply")
PLY_PATH   = MERGED_PLY if os.path.exists(MERGED_PLY) else BASE_PLY

# Output iteration folder
ITER_ID = 0
OUT_DIR = os.path.join(DSL_DIR, "optimize_iteration", f"iter_{ITER_ID:03d}")


def main():
    print("\n[OPT] === Constraint Optimization: same_pairs (midpoint/mean) ===")
    print("[OPT] primitives:", PRIMITIVES_JSON)
    print("[OPT] relations :", RELATIONS_JSON)
    print("[OPT] ply       :", PLY_PATH)
    print("[OPT] out_dir   :", OUT_DIR)

    if not os.path.exists(RELATIONS_JSON):
        raise FileNotFoundError(f"Missing relations.json: {RELATIONS_JSON}")
    if not os.path.exists(PRIMITIVES_JSON):
        raise FileNotFoundError(f"Missing primitives json: {PRIMITIVES_JSON}")
    if not os.path.exists(PLY_PATH):
        raise FileNotFoundError(f"Missing ply: {PLY_PATH}")

    outputs = apply_same_pair_relation(
        primitives_json_path=PRIMITIVES_JSON,
        relations_json_path=RELATIONS_JSON,
        out_dir=OUT_DIR,
        # midpoint behavior:
        use_pointcount_weights=False,  # uniform mean => (A+B)/2 for pairs
        alpha=1.0,                     # apply the mean target directly
        min_same_confidence=0.0,       # raise if you only trust high-confidence same_pairs
    )

    print("\n[OPT] Done.")
    print("[OPT] optimized_primitives:", outputs["optimized_primitives_json"])
    print("[OPT] report              :", outputs["report_json"])

    print("\n[VIS] Starting per-label before/after visualization...")
    print("      (Blue=before, Orange=after; other points black)\n")

    run_before_after_vis_per_label(
        before_primitives_json=PRIMITIVES_JSON,
        after_primitives_json=outputs["optimized_primitives_json"],
        ply_path=PLY_PATH,
    )


if __name__ == "__main__":
    main()
