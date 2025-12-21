#!/usr/bin/env python3
import os

from constraints_optimization.optimize_bbx_size import optimize_bbx_sizes_from_equivalence
from constraints_optimization.vis_bbx_before_after import run_before_after_vis_per_label


# ---------------------------------------------------------------------
# Paths (HARD-CODED)
# ---------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")

CLUSTERS_DIR = os.path.join(SKETCH_ROOT, "clusters")
DSL_DIR      = os.path.join(SKETCH_ROOT, "dsl_optimize")
PROGRAM_DIR  = os.path.join(SKETCH_ROOT, "program")

# ---- inputs ----

# DSL draft (your semantic relations)  <-- UPDATED
DSL_JSON = os.path.join(PROGRAM_DIR, "dsl_draft.json")

# Prefer merged primitives if they exist
MERGED_PRIMITIVES = os.path.join(DSL_DIR, "merged_pca_primitives.json")
BASE_PRIMITIVES   = os.path.join(DSL_DIR, "pca_primitives.json")
PRIMITIVES_JSON   = MERGED_PRIMITIVES if os.path.exists(MERGED_PRIMITIVES) else BASE_PRIMITIVES

# Prefer merged labeled PLY if it exists
MERGED_PLY = os.path.join(DSL_DIR, "merged_labeled_clusters.ply")
BASE_PLY   = os.path.join(CLUSTERS_DIR, "labeled_clusters.ply")
PLY_PATH   = MERGED_PLY if os.path.exists(MERGED_PLY) else BASE_PLY

# ---- outputs ----
ITER_ID = 0
OUT_DIR = os.path.join(DSL_DIR, "optimize_iteration", f"iter_{ITER_ID:03d}")


def main():
    print("\n[OPT] === Bounding Box Size Optimization (Iteration 000) ===")
    print("[OPT] Using primitives:", PRIMITIVES_JSON)
    print("[OPT] Using DSL:", DSL_JSON)
    print("[OPT] Using PLY:", PLY_PATH)
    print("[OPT] Output dir:", OUT_DIR)

    if not os.path.exists(DSL_JSON):
        raise FileNotFoundError(f"Missing DSL draft JSON: {DSL_JSON}")

    if not os.path.exists(PRIMITIVES_JSON):
        raise FileNotFoundError(f"Missing primitives JSON: {PRIMITIVES_JSON}")

    if not os.path.exists(PLY_PATH):
        raise FileNotFoundError(f"Missing PLY: {PLY_PATH}")

    outputs = optimize_bbx_sizes_from_equivalence(
        primitives_json_path=PRIMITIVES_JSON,
        dsl_json_path=DSL_JSON,
        out_dir=OUT_DIR,
        ply_path=PLY_PATH,
        alpha=1.0,                 # fully tie extents inside equivalence groups
        min_group_members=2,
        use_pointcount_weights=True,
        canonical_use_median=True,
    )

    print("\n[OPT] Optimization complete.")
    print("[OPT] Optimized primitives:", outputs["optimized_primitives_json"])
    print("[OPT] Report:", outputs["optimized_report_json"])

    print("\n[VIS] Launching before/after visualization...")
    print("      Blue   = before")
    print("      Orange = after\n")

    run_before_after_vis_per_label(
    before_primitives_json=PRIMITIVES_JSON,
    after_primitives_json=outputs["optimized_primitives_json"],
    ply_path=PLY_PATH,
    )



if __name__ == "__main__":
    main()
