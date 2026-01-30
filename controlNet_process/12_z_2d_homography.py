#!/usr/bin/env python3
# 12_z_2d_homography.py
#
# Main launcher for homography-based mask warping with constraint optimization
#
# Pipeline:
# 1. Compute homographies from 2D bounding box correspondences (homography/homography.py)
# 2. Compute boundary constraints + anchor points (homography/constraints.py)
# 3. Read hierarchy tree (homography/paste_back.py)
# 4. (next) Paste-back / combine translated masks (homography/combine.py)

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from homography import homography
from homography import constraints
from homography import paste_back  # <-- NEW


def main():
    print("=" * 80)
    print("STEP 1: Computing homographies from 2D bounding box correspondences")
    print("=" * 80)
    homography.main()

    print("\n" + "=" * 80)
    print("STEP 2: Computing boundary constraints + anchor points")
    print("=" * 80)
    constraints.main()

    print("\n" + "=" * 80)
    print("STEP 3: Reading hierarchy tree (root anchored, children movable)")
    print("=" * 80)
    paste_back.main()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
