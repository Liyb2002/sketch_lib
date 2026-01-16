#!/usr/bin/env python3
# graph_building/vis.py

import numpy as np
import open3d as o3d
from typing import Dict, Any, List, Set, Tuple


def _make_obb_lines(obb: Dict[str, Any], color=(0, 1, 0)) -> o3d.geometry.LineSet:
    center = np.asarray(obb["center"], dtype=np.float64)
    axes = np.asarray(obb["axes"], dtype=np.float64)   # 3x3, columns are axes
    extents = np.asarray(obb["extents"], dtype=np.float64)

    corners = []
    for dx in (-1, 1):
        for dy in (-1, 1):
            for dz in (-1, 1):
                corners.append(
                    center
                    + dx * axes[:, 0] * extents[0]
                    + dy * axes[:, 1] * extents[1]
                    + dz * axes[:, 2] * extents[2]
                )
    corners = np.asarray(corners, dtype=np.float64)

    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return ls


def _make_sphere(center, radius=0.002, color=(1, 0, 0)) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(np.asarray(center, dtype=np.float64))
    s.paint_uniform_color(color)
    return s


def _build_neighbors(symmetry: Dict[str, Any], attachments: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    neigh: Dict[str, Set[str]] = {}

    # symmetry pairs
    for p in symmetry.get("pairs", []):
        a, b = p["a"], p["b"]
        neigh.setdefault(a, set()).add(b)
        neigh.setdefault(b, set()).add(a)

    # attachments
    for e in attachments:
        a, b = e["a"], e["b"]
        neigh.setdefault(a, set()).add(b)
        neigh.setdefault(b, set()).add(a)

    return neigh


def verify_relations_vis(
    pts: np.ndarray,
    assigned_ids: np.ndarray,
    bboxes_by_name: Dict[str, Any],
    symmetry: Dict[str, Any],
    attachments: List[Dict[str, Any]],
    vis_anchor_points: bool = True,
    anchor_radius: float = 0.002,
):
    """
    For each label:
      - show full point cloud gray
      - label bbox green
      - neighbor bboxes blue (symmetry or attachment)
      - optionally show attachment anchor points as red spheres
    """
    pts = np.asarray(pts, dtype=np.float64)
    assigned_ids = np.asarray(assigned_ids).reshape(-1)

    names = sorted(bboxes_by_name.keys())
    neighbors = _build_neighbors(symmetry, attachments)

    # Pre-index attachment anchors by endpoint label
    anchor_by_label: Dict[str, List[np.ndarray]] = {}
    if vis_anchor_points:
        for e in attachments:
            a, b = e["a"], e["b"]
            aw = np.asarray(e["anchor_world"], dtype=np.float64)
            anchor_by_label.setdefault(a, []).append(aw)
            anchor_by_label.setdefault(b, []).append(aw)

    base_gray = np.array([0.25, 0.25, 0.25], dtype=np.float64)

    for name in names:
        info = bboxes_by_name[name]
        lid = int(info["label_id"])
        mask = (assigned_ids == lid)

        # full shape with highlighted points (optional)
        colors = np.tile(base_gray[None, :], (pts.shape[0], 1))
        colors[mask] = np.array([1.0, 0.2, 0.2], dtype=np.float64)

        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(pts)
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)

        geoms = [pcd_vis]

        # current bbox green
        geoms.append(_make_obb_lines(info["obb_pca"], color=(0, 1, 0)))

        # neighbor bboxes blue
        for nb in sorted(neighbors.get(name, set())):
            if nb not in bboxes_by_name:
                continue
            geoms.append(_make_obb_lines(bboxes_by_name[nb]["obb_pca"], color=(0, 0, 1)))

        # anchors (red spheres)
        if vis_anchor_points:
            for aw in anchor_by_label.get(name, []):
                geoms.append(_make_sphere(aw, radius=anchor_radius, color=(1, 0, 0)))

        title = f"[VERIFY] {name}  (neighbors={len(neighbors.get(name, set()))})"
        print(title)
        o3d.visualization.draw_geometries(geoms, window_name=title)
