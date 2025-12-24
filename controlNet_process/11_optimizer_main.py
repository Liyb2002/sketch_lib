#!/usr/bin/env python3
# 11_optimizer_main.py
#
# Pre-optimization debug:
#   For each label:
#     - show saved heatmap PLY (already generated elsewhere)
#     - overlay bounding boxes read the SAME WAY as no_overlapping.py:
#         mn = center - 0.5*extent, mx = center + 0.5*extent
#
# IMPORTANT:
#   - never imports or calls constraints_optimization.heat_map / build_label_heatmaps

import os

from constraints_optimization.vis_preopt_label_boxes_heatmap import (
    vis_preopt_label_boxes_heatmap,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")

DSL_DIR = os.path.join(SKETCH_ROOT, "dsl_optimize")

MERGED_PRIMITIVES = os.path.join(DSL_DIR, "merged_pca_primitives.json")
BASE_PRIMITIVES   = os.path.join(DSL_DIR, "pca_primitives.json")
PRIMITIVES_JSON   = MERGED_PRIMITIVES if os.path.exists(MERGED_PRIMITIVES) else BASE_PRIMITIVES

ITER_ID = 0
OUT_DIR  = os.path.join(DSL_DIR, "optimize_iteration", f"iter_{ITER_ID:03d}")
HEAT_DIR = os.path.join(OUT_DIR, "heat_map")


def main():
    print("\n[PRE-OPT VIS] === per-label heatmap + AABB boxes (optimizer-style) ===")
    print("[PRE-OPT VIS] primitives :", PRIMITIVES_JSON)
    print("[PRE-OPT VIS] heat_dir   :", HEAT_DIR)

    if not os.path.exists(PRIMITIVES_JSON):
        raise FileNotFoundError(f"[FATAL] Missing primitives json: {PRIMITIVES_JSON}")

    summary_json = os.path.join(HEAT_DIR, "heatmaps_summary.json")
    if not os.path.exists(summary_json):
        raise FileNotFoundError(
            f"[FATAL] Missing heatmaps_summary.json:\n  {summary_json}\n"
            "Heatmaps must already exist. This script does NOT generate them."
        )

    vis_preopt_label_boxes_heatmap(
        primitives_json_path=PRIMITIVES_JSON,
        heat_dir=HEAT_DIR,
        max_labels=None,
        max_boxes_per_label=None,
    )


if __name__ == "__main__":
    main()
