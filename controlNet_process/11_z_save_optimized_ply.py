#!/usr/bin/env python3
"""
launcher_save_optimized_components_ply.py

Reads:
  sketch/dsl_optimize/optimize_iteration/iter_000/heat_map/pca_bboxes/pca_bboxes_optimized.json

For each label:
  - loads its heatmap PLY (entry["heat_ply"])
  - keeps ALL points, but recolors points OUTSIDE the chosen AABB to BLACK
    (opt_aabb_world preferred; fallback to aabb min_bound/max_bound)
  - saves a per-label .ply into:
      sketch/dsl_optimize/optimize_iteration/iter_000/optimized_components/<label>.ply
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)

from constraints_optimization.save_new_segmentation import save_optimized_component_plys

SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")
DSL_DIR     = os.path.join(SKETCH_ROOT, "dsl_optimize")

ITER_ID = 0
OUT_DIR = os.path.join(DSL_DIR, "optimize_iteration", f"iter_{ITER_ID:03d}")

HEAT_DIR = os.path.join(OUT_DIR, "heat_map")
PCA_BBOX_OPT_JSON = os.path.join(HEAT_DIR, "pca_bboxes", "pca_bboxes_optimized.json")

OPT_COMPONENT_DIR = os.path.join(OUT_DIR, "optimized_components")


def main():
    if not os.path.isfile(PCA_BBOX_OPT_JSON):
        raise FileNotFoundError(f"Missing: {PCA_BBOX_OPT_JSON}")

    print("[LAUNCH] pca_bboxes_optimized:", PCA_BBOX_OPT_JSON)
    print("[LAUNCH] out_dir            :", OPT_COMPONENT_DIR)

    save_optimized_component_plys(
        pca_bboxes_optimized_json=PCA_BBOX_OPT_JSON,
        out_dir=OPT_COMPONENT_DIR,
        prefer_opt_aabb=True,
        overwrite=True,
    )


if __name__ == "__main__":
    main()
