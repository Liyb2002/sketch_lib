#!/usr/bin/env python3
"""
launcher_heatmap_pca_vis.py

1) Compute PCA bboxes from saved heatmaps
2) (Optional) visualize
3) Optimize bboxes (shrink-only) to reduce overlap with minimal value loss
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)

from constraints_optimization.pca_analysis import compute_pca_bounding_boxes
from constraints_optimization.vis import visualize_heatmaps_with_bboxes
from constraints_optimization.optimizer import optimize_bounding_boxes
from constraints_optimization.vis import visualize_heatmaps_with_bboxes_before_after


SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")
DSL_DIR     = os.path.join(SKETCH_ROOT, "dsl_optimize")

ITER_ID = 0
OUT_DIR = os.path.join(DSL_DIR, "optimize_iteration", f"iter_{ITER_ID:03d}")

HEAT_DIR  = os.path.join(OUT_DIR, "heat_map")
BBOX_DIR  = os.path.join(HEAT_DIR, "pca_bboxes")
BBOX_JSON = os.path.join(BBOX_DIR, "pca_bboxes.json")

OPT_BBOX_JSON = os.path.join(BBOX_DIR, "pca_bboxes_optimized.json")
NO_OVERLAP_REPORT = os.path.join(OUT_DIR, "no_overlapping_report.json")


def main():
    if not os.path.isdir(HEAT_DIR):
        raise FileNotFoundError(
            f"Missing heat map dir: {HEAT_DIR}\n"
            f"Did you run the heatmap step first?"
        )

    print("\n[LAUNCH] === Step 1: PCA BBOX from saved heatmaps ===")
    print("[LAUNCH] heat_dir :", HEAT_DIR)
    print("[LAUNCH] bbox_json:", BBOX_JSON)

    compute_pca_bounding_boxes(
        heat_dir=HEAT_DIR,
        out_json=BBOX_JSON,
        min_heat=0.5,
        min_points=200,
        max_labels=None,
    )

    # (Optional) visualize original
    # print("\n[LAUNCH] === Step 2: VIS (original bboxes) ===")
    # visualize_heatmaps_with_bboxes(
    #     heat_dir=HEAT_DIR,
    #     bbox_json=BBOX_JSON,
    #     max_labels_to_show=12,
    #     darken_heatmap=0.7,
    #     bbox_radius=0.003,
    #     print_grid_values=True,
    #     grid_res=3,
    # )

    # Step 3: optimize (shrink-only)
    print("\n[LAUNCH] === Step 3: Optimize bboxes (shrink-only, value-aware) ===")
    optimize_bounding_boxes(
        bbox_json=BBOX_JSON,
        out_optimized_bbox_json=OPT_BBOX_JSON,
        out_report_json=NO_OVERLAP_REPORT,
        max_iter=200,
        step_frac=0.06,
        min_extent_frac=0.15,
        w_overlap=1.0,
        w_value=1.0,
        w_same=0.0,  # accepted but ignored (backward compat)
        verbose=True,
    )

    # (Optional) visualize optimized
    visualize_heatmaps_with_bboxes_before_after(
        heat_dir=HEAT_DIR,
        bbox_json_before=BBOX_JSON,
        bbox_json_after=OPT_BBOX_JSON,
        max_labels_to_show=12,
    )


if __name__ == "__main__":
    main()
