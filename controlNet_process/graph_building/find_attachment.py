#!/usr/bin/env python3
# graph_building/find_attachment.py

import numpy as np
from typing import Dict, Any, List, Tuple


def _world_to_object(p_world: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    # axes columns are object axes
    return (p_world - origin) @ axes


def _object_to_world(p_local: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return origin + axes @ p_local


def _aabb_minmax_in_object_space(obb: Dict[str, Any], origin: np.ndarray, axes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Our OBBs are aligned to the object axes already. extents are half-sizes along axes.
    So in object coordinates, box is axis-aligned:
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


def _aabb_distance_and_closest_points(mn1, mx1, mn2, mx2) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Returns:
      dist, p1_local, p2_local

    For each axis:
      - If intervals overlap: closest points share any value in overlap (we use overlap midpoint)
      - Else: closest points are nearest faces (max1 vs min2, or max2 vs min1)
    """
    p1 = np.zeros(3, dtype=np.float64)
    p2 = np.zeros(3, dtype=np.float64)

    for k in range(3):
        if mx1[k] < mn2[k]:
            # box1 is "left" of box2
            p1[k] = mx1[k]
            p2[k] = mn2[k]
        elif mx2[k] < mn1[k]:
            # box2 is "left" of box1
            p1[k] = mn1[k]
            p2[k] = mx2[k]
        else:
            # overlap: choose midpoint of overlap interval
            lo = max(mn1[k], mn2[k])
            hi = min(mx1[k], mx2[k])
            mid = 0.5 * (lo + hi)
            p1[k] = mid
            p2[k] = mid

    d = float(np.linalg.norm(p1 - p2))
    return d, p1, p2


def find_attachments(
    bboxes_by_name: Dict[str, Any],
    object_space: Dict[str, Any],
    attach_thresh: float,
    ignore_unknown: bool = False,
) -> List[Dict[str, Any]]:
    """
    Args:
      bboxes_by_name: dict like the loaded label_bboxes_pca.json
        bboxes_by_name[name]["obb_pca"] = {center, axes, extents}
      object_space: {"origin":[...], "axes":[[...],[...],[...]]} (axes columns)
      attach_thresh: distance threshold in OBJECT SPACE units
      ignore_unknown: if True, skip any edge with unknown_*

    Returns:
      list of edges:
        {
          "a": name_i,
          "b": name_j,
          "distance": float,
          "anchor_local": [x,y,z],
          "anchor_world": [x,y,z]
        }
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
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]

            if ignore_unknown and (a.startswith("unknown_") or b.startswith("unknown_")):
                continue

            mn1, mx1 = aabbs[a]
            mn2, mx2 = aabbs[b]

            dist, p1, p2 = _aabb_distance_and_closest_points(mn1, mx1, mn2, mx2)
            if dist <= attach_thresh:
                anchor_local = 0.5 * (p1 + p2)
                anchor_world = _object_to_world(anchor_local, origin, axes)
                edges.append({
                    "a": a,
                    "b": b,
                    "distance": dist,
                    "anchor_local": anchor_local.tolist(),
                    "anchor_world": anchor_world.tolist(),
                })

    return edges
