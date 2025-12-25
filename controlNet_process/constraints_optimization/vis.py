#!/usr/bin/env python3
"""
constraints_optimization/vis.py

Visualization:
- Load heatmap PLY (colored full-shape point cloud)
- Overlay PCA OBB as THICK BLUE BOUNDARY ONLY
  (cylinders per edge, using Open3D LineSet connectivity to avoid wrong corner ordering)

No solid boxes.
"""

import json
from typing import Any, Dict, List

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _load_colored_ply(path: str) -> "o3d.geometry.PointCloud":
    if o3d is None:
        raise RuntimeError("open3d required")
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pcd


# ---------------------------------------------------------------------
# OBB restore
# ---------------------------------------------------------------------

def _obb_from_dict(d: Dict[str, Any]) -> "o3d.geometry.OrientedBoundingBox":
    if o3d is None:
        raise RuntimeError("open3d required")
    return o3d.geometry.OrientedBoundingBox(
        center=np.asarray(d["center"], dtype=np.float64),
        R=np.asarray(d["R"], dtype=np.float64),
        extent=np.asarray(d["extent"], dtype=np.float64),
    )


# ---------------------------------------------------------------------
# Thick boundary rendering (cylinders per *correct* LineSet edge)
# ---------------------------------------------------------------------

def _cylinder_between(p0: np.ndarray, p1: np.ndarray, radius: float, color) -> "o3d.geometry.TriangleMesh":
    """
    Create a cylinder mesh between two 3D points, colored uniformly.
    Cylinder is aligned from p0 to p1.
    """
    v = p1 - p0
    length = float(np.linalg.norm(v))
    if length < 1e-10:
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=float(radius), height=length)
    cyl.compute_vertex_normals()

    # Open3D cylinder axis is +Z. Rotate Z -> v.
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v_hat = v / length

    axis = np.cross(z, v_hat)
    axis_norm = float(np.linalg.norm(axis))
    dot = float(np.clip(np.dot(z, v_hat), -1.0, 1.0))
    angle = float(np.arccos(dot))

    if axis_norm > 1e-10 and angle > 1e-10:
        axis = axis / axis_norm
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cyl.rotate(R, center=(0.0, 0.0, 0.0))

    # Move cylinder to midpoint of segment
    cyl.translate((p0 + p1) * 0.5)
    cyl.paint_uniform_color(color)
    return cyl


def _thick_obb_boundary_from_lineset(
    obb: "o3d.geometry.OrientedBoundingBox",
    *,
    radius: float,
    color=(0.0, 0.0, 1.0),  # BLUE
) -> List["o3d.geometry.TriangleMesh"]:
    """
    Use Open3D's own LineSet connectivity (guaranteed correct),
    then replace each line with a thick cylinder.
    """
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    pts = np.asarray(ls.points, dtype=np.float64)
    lines = np.asarray(ls.lines, dtype=np.int64)

    meshes = []
    for a, b in lines:
        m = _cylinder_between(pts[a], pts[b], radius=radius, color=color)
        if m is not None:
            meshes.append(m)
    return meshes


# ---------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------

def visualize_heatmaps_with_bboxes(
    *,
    heat_dir: str,                 # kept for compatibility with your launcher; not used internally
    bbox_json: str,
    max_labels_to_show: int = 12,
    darken_heatmap: float = 0.7,
    bbox_radius: float = 0.003,     # thickness control
) -> None:
    """
    For each label:
      - show heatmap
      - overlay thick BLUE bbox boundary

    bbox_radius controls thickness in world units.
    """
    if o3d is None:
        raise RuntimeError("open3d required")

    payload = _load_json(bbox_json)
    labels = payload.get("labels", [])
    if not labels:
        print("[VIS] No labels to visualize.")
        return

    labels = sorted(labels, key=lambda r: int(r.get("points_used", 0)), reverse=True)
    labels = labels[: int(max_labels_to_show)]

    for rec in labels:
        label = rec.get("label", rec.get("sanitized", "unknown"))
        pcd = _load_colored_ply(rec["heat_ply"])

        # Darken heatmap so bbox pops
        if float(darken_heatmap) < 0.999:
            cols = np.asarray(pcd.colors)
            cols = np.clip(cols * float(darken_heatmap), 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(cols)

        obb = _obb_from_dict(rec["obb"])

        bbox_meshes = _thick_obb_boundary_from_lineset(
            obb,
            radius=float(bbox_radius),
            color=(0.0, 0.0, 1.0),  # BLUE fixed
        )

        geoms = [pcd] + bbox_meshes
        title = f"HeatMap + PCA BBox (BLUE) | {label} | used={rec.get('points_used')} | min_heat={rec.get('min_heat')}"
        print("[VIS] opening:", title)
        o3d.visualization.draw_geometries(geoms, window_name=title)