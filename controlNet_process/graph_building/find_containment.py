#!/usr/bin/env python3
# graph_building/find_containment.py

import numpy as np
from typing import Dict, Any, List, Tuple


def _world_to_object(p_world: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return (p_world - origin) @ axes


def _aabb_minmax_in_object_space(obb: Dict[str, Any], origin: np.ndarray, axes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Since our OBBs are aligned to object axes, in object coords the box is an AABB:
      center_local = (center_world - origin) @ axes
      min = center_local - extents
      max = center_local + extents
    """
    c_world = np.array(obb["center"], dtype=np.float64)
    e = np.array(obb["extents"], dtype=np.float64)
    c_local = _world_to_object(c_world, origin, axes)
    mn = c_local - e
    mx = c_local + e
    return mn, mx


def find_containment(
    bboxes_by_name: Dict[str, Any],
    object_space: Dict[str, Any],
    contain_tol: float = 1e-6,
    ignore_unknown: bool = False,
    require_strict: bool = True,
    strict_margin: float = 1e-4,
) -> List[Dict[str, Any]]:
    """
    Returns containment edges:
      {
        "outer": A,
        "inner": B,
        "margin_min": [dx,dy,dz],   # how much room on min side (A_min - B_min) negative means violation
        "margin_max": [dx,dy,dz],   # how much room on max side (B_max - A_max) negative means violation
      }

    Containment test in object space:
      A contains B if:
        A_min <= B_min + tol  and  B_max <= A_max + tol   (elementwise)

    If require_strict:
      also require B be meaningfully smaller than A on at least one axis by `strict_margin`
      to avoid "mutual containment" on identical boxes.
    """
    origin = np.array(object_space["origin"], dtype=np.float64)
    axes = np.array(object_space["axes"], dtype=np.float64)

    names = sorted(bboxes_by_name.keys())
    aabbs = {}

    for n in names:
        obb = bboxes_by_name[n]["obb_pca"]
        mn, mx = _aabb_minmax_in_object_space(obb, origin, axes)
        aabbs[n] = (mn, mx)

    edges: List[Dict[str, Any]] = []

    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue

            A = names[i]
            B = names[j]

            if ignore_unknown and (A.startswith("unknown_") or B.startswith("unknown_")):
                continue

            mnA, mxA = aabbs[A]
            mnB, mxB = aabbs[B]

            # margins: positive means B is inside with that slack (approx)
            margin_min = (mnB - mnA)  # B_min - A_min
            margin_max = (mxA - mxB)  # A_max - B_max

            inside = np.all(mnA <= (mnB + contain_tol)) and np.all(mxB <= (mxA + contain_tol))
            if not inside:
                continue

            if require_strict:
                # require B strictly smaller than A on at least one axis by strict_margin
                sizeA = mxA - mnA
                sizeB = mxB - mnB
                if not np.any((sizeA - sizeB) > strict_margin):
                    continue

            edges.append({
                "outer": A,
                "inner": B,
                "margin_min": margin_min.tolist(),
                "margin_max": margin_max.tolist(),
            })

    return edges
