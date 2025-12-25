#!/usr/bin/env python3
"""
launcher_heatmap_pca_vis.py

1) Load saved per-label heatmap PLYs from:
   sketch/dsl_optimize/optimize_iteration/iter_XXX/heat_map/heatmaps/<label>/heat_map_<label>.ply

2) Compute PCA-oriented bounding boxes (Open3D OrientedBoundingBox) per label
   using points whose heat >= min_heat (default 0.5).

3) Visualize (per label): heatmap point cloud + bbox overlay.

Writes:
  .../iter_XXX/heat_map/pca_bboxes/pca_bboxes.json
"""

import os
import sys

# Make sure we can import constraints_optimization/*
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)

from constraints_optimization.pca_analysis import compute_pca_bounding_boxes
from constraints_optimization.vis import visualize_heatmaps_with_bboxes


# -----------------------------------------------------------------------------
# Paths (match your existing pipeline style)
# -----------------------------------------------------------------------------
SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")
CLUSTERS_DIR = os.path.join(SKETCH_ROOT, "clusters")
DSL_DIR      = os.path.join(SKETCH_ROOT, "dsl_optimize")

# Which iteration to use
ITER_ID = 0
OUT_DIR = os.path.join(DSL_DIR, "optimize_iteration", f"iter_{ITER_ID:03d}")

HEAT_DIR = os.path.join(OUT_DIR, "heat_map")  # where your heat_map.py writes
BBOX_DIR = os.path.join(HEAT_DIR, "pca_bboxes")
BBOX_JSON = os.path.join(BBOX_DIR, "pca_bboxes.json")


def main():
    if not os.path.isdir(HEAT_DIR):
        raise FileNotFoundError(
            f"Missing heat map dir: {HEAT_DIR}\n"
            f"Did you run the heatmap step first?"
        )

    print("\n[LAUNCH] === PCA BBOX + VIS from saved heatmaps ===")
    print("[LAUNCH] heat_dir :", HEAT_DIR)
    print("[LAUNCH] bbox_json:", BBOX_JSON)

    # 1) Compute PCA bboxes from saved heatmap PLYs
    compute_pca_bounding_boxes(
        heat_dir=HEAT_DIR,
        out_json=BBOX_JSON,
        min_heat=0.5,              # points with heat>=0.5 define the label bbox (tune if needed)
        min_points=200,            # skip labels with too few hot points
        max_labels=None,           # None => all
    )

    # 2) Visualize per label: heatmap + bbox overlay
    visualize_heatmaps_with_bboxes(
        heat_dir=HEAT_DIR,
        bbox_json=BBOX_JSON,
        max_labels_to_show=12,     # matches your previous style
    )


if __name__ == "__main__":
    main()