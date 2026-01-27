#!/usr/bin/env python3
# AEP/find_next_face/vis_next_face.py
#
# Visualization for next face edit change
# Color scheme: ALL BEFORE = BLUE, ALL AFTER = RED, chosen face = GREEN

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import os
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _axes_cols(axes_list: Any) -> np.ndarray:
    """Treat JSON axes as COLUMNS."""
    R = _as_np(axes_list)
    if R.shape != (3, 3):
        raise ValueError(f"axes must be 3x3, got {R.shape}")
    return R


def _make_lineset_from_obb(obb: Dict[str, Any]) -> "o3d.geometry.LineSet":
    if o3d is None:
        raise RuntimeError("open3d is required for visualization but is not available.")
    C = _as_np(obb["center"])
    R = _axes_cols(obb["axes"])
    E = _as_np(obb["extents"])
    obb_o3d = o3d.geometry.OrientedBoundingBox(center=C, R=R, extent=2.0 * E)
    return o3d.geometry.LineSet.create_from_oriented_bounding_box(obb_o3d)


def _color_geom(geom, rgb: Tuple[float, float, float]):
    if hasattr(geom, "paint_uniform_color"):
        geom.paint_uniform_color(list(rgb))
    return geom


def _obb_face_corners_world(obb: Dict[str, Any], axis: int, sign: int) -> np.ndarray:
    C = _as_np(obb["center"])
    R = _axes_cols(obb["axes"])
    E = _as_np(obb["extents"])

    a = int(axis)
    s = +1 if int(sign) >= 0 else -1

    idx = [0, 1, 2]
    idx.remove(a)
    b, c = idx[0], idx[1]

    q_list = []
    for sb in (-1.0, +1.0):
        for sc in (-1.0, +1.0):
            q = np.zeros(3, dtype=np.float64)
            q[a] = s * float(E[a])
            q[b] = sb * float(E[b])
            q[c] = sc * float(E[c])
            q_list.append(q)

    Q = np.stack(q_list, axis=0)   # (4,3)
    P = C[None, :] + (Q @ R.T)
    return P


def _make_face_patch(
    obb: Dict[str, Any],
    axis: int,
    sign: int,
    color_rgb: Tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> Tuple["o3d.geometry.TriangleMesh", "o3d.geometry.LineSet"]:
    if o3d is None:
        raise RuntimeError("open3d is required for visualization but is not available.")

    P = _obb_face_corners_world(obb, axis=axis, sign=sign)
    pts = np.array([P[0], P[1], P[3], P[2]], dtype=np.float64)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts)
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    mesh.compute_vertex_normals()
    _color_geom(mesh, color_rgb)

    border = o3d.geometry.LineSet()
    border.points = o3d.utility.Vector3dVector(pts)
    border.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32))
    _color_geom(border, color_rgb)

    return mesh, border


def vis_next_face_edit_change(
    face_edit_change: Dict[str, Any],
    overlay_ply_path: Optional[str] = None,
    show_overlay: bool = True,
    window_name: Optional[str] = None,
):
    """
    Visualization:
      - overlay PLY in GREY
      - ALL BEFORE boxes: BLUE (target before + neighbor before)
      - ALL AFTER boxes: RED (target after + neighbor after)
      - chosen face highlight on neighbor after (GREEN patch + border)
    """
    if o3d is None:
        return

    ch = face_edit_change.get("change", {}) or {}

    nb_before = ch.get("before_obb", None)
    nb_after = ch.get("after_obb", None)

    face = ch.get("face", None)
    axis = int(ch.get("axis", 0))
    sign = int(ch.get("sign", +1))

    input_dbg = ch.get("input_edit_debug", {}) or {}
    t_before = input_dbg.get("target_edit_before_obb", None)
    t_after = input_dbg.get("target_edit_after_obb", None)

    if not isinstance(nb_before, dict) or not isinstance(nb_after, dict) or not isinstance(face, str):
        return

    geoms = []

    # Overlay (GREY)
    if show_overlay and overlay_ply_path and os.path.isfile(overlay_ply_path):
        try:
            ply = o3d.io.read_point_cloud(overlay_ply_path)
            ply = _color_geom(ply, (0.6, 0.6, 0.6))
            geoms.append(ply)
        except Exception:
            pass

    # ALL BEFORE boxes: BLUE
    if isinstance(t_before, dict):
        tls_before = _make_lineset_from_obb(t_before)
        _color_geom(tls_before, (0.0, 0.0, 1.0))  # BLUE
        geoms.append(tls_before)
    
    ls_nb_before = _make_lineset_from_obb(nb_before)
    _color_geom(ls_nb_before, (0.0, 0.0, 1.0))  # BLUE
    geoms.append(ls_nb_before)

    # ALL AFTER boxes: RED
    if isinstance(t_after, dict):
        tls_after = _make_lineset_from_obb(t_after)
        _color_geom(tls_after, (1.0, 0.0, 0.0))  # RED
        geoms.append(tls_after)
    
    ls_nb_after = _make_lineset_from_obb(nb_after)
    _color_geom(ls_nb_after, (1.0, 0.0, 0.0))  # RED
    geoms.append(ls_nb_after)

    # Chosen face highlight on neighbor AFTER (GREEN)
    try:
        face_mesh, face_border = _make_face_patch(nb_after, axis=axis, sign=sign, color_rgb=(0.0, 1.0, 0.0))
        geoms.append(face_mesh)
        geoms.append(face_border)
    except Exception:
        pass

    title = window_name or f"next_face_edit_change | target={face_edit_change.get('target')} face={face}"
    try:
        o3d.visualization.draw_geometries(geoms, window_name=title)
    except TypeError:
        o3d.visualization.draw_geometries(geoms)