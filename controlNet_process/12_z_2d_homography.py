#!/usr/bin/env python3
# 12_z_2d_homography.py
#
# Main launcher for homography-based mask warping with constraint optimization
# 
# Pipeline:
# 1. Compute homographies from 2D bounding box correspondences (homography/homography.py)
# 2. Optimize component positions to respect boundary constraints (homography/constraints.py)
#
# Inputs:
#   sketch/back_project_masks/view_{x}/3d_project/obb_2d_projections.json
#   sketch/hierarchy_tree.json
#
# Outputs:
#   sketch/back_project_masks/view_{x}/homography/... (homography results)
#   sketch/back_project_masks/view_{x}/constraints/... (constraint-optimized results)

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Import the two processing modules
from homography import homography
from homography import constraints


def main():
    print("=" * 80)
    print("STEP 1: Computing homographies from 2D bounding box correspondences")
    print("=" * 80)
    homography.main()
    
    print("\n" + "=" * 80)
    print("STEP 2: Optimizing component positions with boundary constraints")
    print("=" * 80)
    constraints.main()
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()