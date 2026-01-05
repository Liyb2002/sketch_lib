#!/usr/bin/env python3
"""
launcher_heatmap_vis_only.py

Just loads each per-label heatmap PLY under:
  sketch/dsl_optimize/optimize_iteration/iter_000/heat_map/heatmaps/<label>/heat_map_*.ply

and visualizes them (one by one).
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)

from constraints_optimization.save_new_segmentation import vis_heatmap_plys_per_label


SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")
DSL_DIR     = os.path.join(SKETCH_ROOT, "dsl_optimize")

ITER_ID = 0
OUT_DIR = os.path.join(DSL_DIR, "optimize_iteration", f"iter_{ITER_ID:03d}")

HEAT_DIR = os.path.join(OUT_DIR, "heat_map")


def main():
    if not os.path.isdir(HEAT_DIR):
        raise FileNotFoundError(f"Missing heat map dir: {HEAT_DIR}")

    print("[LAUNCH] heat_dir:", HEAT_DIR)
    vis_heatmap_plys_per_label(heat_dir=HEAT_DIR)


if __name__ == "__main__":
    main()
