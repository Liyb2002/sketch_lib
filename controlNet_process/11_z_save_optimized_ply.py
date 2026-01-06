#!/usr/bin/env python3
"""
launcher_vis_all_heatmaps_plus_opt_aabb.py

For each label entry in:
  sketch/dsl_optimize/optimize_iteration/iter_000/heat_map/pca_bboxes/pca_bboxes_optimized.json

Visualize:
  - the per-label heatmap point cloud (entry["heat_ply"])
  - the optimized world AABB (entry["opt_aabb_world"]) if present
    (fallback to entry["aabb"] if opt_aabb_world missing)

If the heatmap PLY does not exist, print:
  [MISS] <label> : heatmap ply not found at <path>
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)

from constraints_optimization.save_new_segmentation import vis_all_labels_heatmap_with_opt_aabb

SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")
DSL_DIR     = os.path.join(SKETCH_ROOT, "dsl_optimize")

ITER_ID = 0
OUT_DIR = os.path.join(DSL_DIR, "optimize_iteration", f"iter_{ITER_ID:03d}")
HEAT_DIR = os.path.join(OUT_DIR, "heat_map")

PCA_BBOX_DIR = os.path.join(HEAT_DIR, "pca_bboxes")
PCA_BBOX_OPT_JSON = os.path.join(PCA_BBOX_DIR, "pca_bboxes_optimized.json")


def main():
    if not os.path.isfile(PCA_BBOX_OPT_JSON):
        raise FileNotFoundError(f"Missing: {PCA_BBOX_OPT_JSON}")

    print("[LAUNCH] pca_bboxes_optimized:", PCA_BBOX_OPT_JSON)

    vis_all_labels_heatmap_with_opt_aabb(
        pca_bboxes_optimized_json=PCA_BBOX_OPT_JSON,
        show_obb=False,
        prefer_opt_aabb=True,
    )


if __name__ == "__main__":
    main()
