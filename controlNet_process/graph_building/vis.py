#!/usr/bin/env python3
# graph_building/vis.py

import os
import json
import numpy as np
import open3d as o3d
from typing import Dict, Any, List, Tuple, Optional


# ----------------------------
# Helpers: bbox parsing / transforms
# ----------------------------

def _get_obb_pca(bbox_entry: Dict[str, Any]) -> Dict[str, Any]:
    if "obb_pca" not in bbox_entry:
        raise KeyError("bbox entry missing 'obb_pca'")
    obb = bbox_entry["obb_pca"]
    if "center" not in obb or "extents" not in obb or "axes" not in obb:
        raise KeyError("obb_pca missing 'center'/'extents'/'axes'")
    return obb


def _world_to_object(p_world: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    # axes columns are object axes
    return (p_world - origin) @ axes


def _object_to_world(p_local: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    # axes columns are object axes
    return origin + axes @ p_local


def _obb_from_entry(entry: Dict[str, Any]) -> o3d.geometry.OrientedBoundingBox:
    obb = _get_obb_pca(entry)
    center = np.array(obb["center"], dtype=np.float64)
    axes = np.array(obb["axes"], dtype=np.float64)
    extents = np.array(obb["extents"], dtype=np.float64) * 2.0  # open3d expects full lengths
    if axes.shape != (3, 3):
        raise ValueError("obb_pca['axes'] must be 3x3")
    return o3d.geometry.OrientedBoundingBox(center=center, R=axes, extent=extents)


def _lineset_from_obb(obb: o3d.geometry.OrientedBoundingBox, color=(0.0, 0.0, 0.0)) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    ls.paint_uniform_color(color)
    return ls


def _sphere(center: np.ndarray, radius: float, color=(1.0, 0.0, 0.0)) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
    s.translate(center.astype(np.float64))
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    return s


def _colorize_assigned_ids(assigned_ids: np.ndarray, highlight_mask: np.ndarray, neighbor_mask: np.ndarray) -> np.ndarray:
    """
    Return (N,3) colors:
      - background: light gray
      - neighbors: orange
      - highlight (label): blue
      - unknowns: darker gray
    """
    N = assigned_ids.shape[0]
    colors = np.zeros((N, 3), dtype=np.float64)
    colors[:] = np.array([0.75, 0.75, 0.75], dtype=np.float64)

    # neighbor
    colors[neighbor_mask] = np.array([1.0, 0.6, 0.1], dtype=np.float64)
    # highlight
    colors[highlight_mask] = np.array([0.1, 0.3, 1.0], dtype=np.float64)

    # (optional) unknown id = -1 -> dark
    colors[assigned_ids < 0] = np.array([0.4, 0.4, 0.4], dtype=np.float64)
    return colors


def _get_name_to_id(bboxes_by_name: Dict[str, Any]) -> Dict[str, int]:
    out = {}
    for name, entry in bboxes_by_name.items():
        if isinstance(entry, dict) and "label_id" in entry:
            out[name] = int(entry["label_id"])
    return out


def _attachments_for_label(attachments: List[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    out = []
    for e in attachments:
        if e.get("a") == name or e.get("b") == name:
            out.append(e)
    return out


def _neighbor_names(attachments: List[Dict[str, Any]], name: str) -> List[str]:
    neigh = set()
    for e in attachments:
        a = e.get("a")
        b = e.get("b")
        if a == name and isinstance(b, str):
            neigh.add(b)
        elif b == name and isinstance(a, str):
            neigh.add(a)
    return sorted(list(neigh))


def _other_name_in_edge(e: Dict[str, Any], name: str) -> Optional[str]:
    a = e.get("a", None)
    b = e.get("b", None)
    if a == name and isinstance(b, str):
        return b
    if b == name and isinstance(a, str):
        return a
    return None


# ----------------------------
# Relation type inference (robust to older/newer formats)
# ----------------------------

def _infer_attachment_kind(e: Dict[str, Any]) -> str:
    """
    Returns one of: "volume", "face", "point"
    Priority:
      1) explicit keys if present
      2) volume metrics
      3) face keys
      4) fallback -> point
    """
    for k in ["kind", "attachment_kind", "relation_kind", "attachment_type", "relation_type"]:
        v = e.get(k, None)
        if isinstance(v, str):
            vv = v.lower()
            if "vol" in vv:
                return "volume"
            if "face" in vv:
                return "face"
            if "point" in vv or "anchor" in vv:
                return "point"

    # heuristics
    if any(x in e for x in ["overlap_volume", "vol_overlap", "overlap_box_local_min", "overlap_box_local_max"]):
        return "volume"
    if ("a_face" in e) and ("b_face" in e):
        return "face"
    return "point"


def _neighbor_names_for_kind(att_edges: List[Dict[str, Any]], name: str, kind: str) -> List[str]:
    """
    For a given label 'name', return the neighbor labels that participate in attachments of 'kind'.
    """
    out = set()
    for e in att_edges:
        if _infer_attachment_kind(e) != kind:
            continue
        other = _other_name_in_edge(e, name)
        if other is not None:
            out.add(other)
    return sorted(list(out))


# ----------------------------
# Volume overlap visualization (approx)
# ----------------------------

def _obb_corners_world(obb: o3d.geometry.OrientedBoundingBox) -> np.ndarray:
    return np.asarray(obb.get_box_points(), dtype=np.float64)


def _aabb_minmax_in_object_from_obb(
    obb: o3d.geometry.OrientedBoundingBox,
    origin: np.ndarray,
    obj_axes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    corners_w = _obb_corners_world(obb)                # (8,3)
    corners_l = _world_to_object(corners_w, origin, obj_axes)  # (8,3)
    mn = corners_l.min(axis=0)
    mx = corners_l.max(axis=0)
    return mn, mx


def _intersection_aabb(
    mn1: np.ndarray,
    mx1: np.ndarray,
    mn2: np.ndarray,
    mx2: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    mn = np.maximum(mn1, mn2)
    mx = np.minimum(mx1, mx2)
    d = mx - mn
    if np.any(d <= 0.0):
        return None, None, 0.0
    vol = float(d[0] * d[1] * d[2])
    return mn, mx, vol


def _aabb_mesh_in_object_space(
    mn: np.ndarray,
    mx: np.ndarray,
    origin: np.ndarray,
    obj_axes: np.ndarray,
    color=(1.0, 0.0, 0.0)
) -> o3d.geometry.TriangleMesh:
    center_l = (mn + mx) / 2.0
    extent = (mx - mn)
    center_w = _object_to_world(center_l, origin, obj_axes)
    obb = o3d.geometry.OrientedBoundingBox(center=center_w, R=obj_axes, extent=extent)
    mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obb)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


# ----------------------------
# Face visualization (object-space faces +u0/-u0 style)
# ----------------------------

def _face_id_to_axis_sign(face_id: str) -> Tuple[int, float]:
    # "+u0" -> (0, +1), "-u2" -> (2, -1)
    if not isinstance(face_id, str) or len(face_id) < 3:
        return 0, +1.0
    sign = +1.0 if face_id[0] == "+" else -1.0
    k = int(face_id[-1])
    return k, sign


def _quad_mesh_for_face_in_object_aabb(
    mn_l: np.ndarray,
    mx_l: np.ndarray,
    face_id: str,
    origin: np.ndarray,
    obj_axes: np.ndarray,
    thickness: float = 1e-4,
    color=(0.0, 0.2, 1.0),
) -> o3d.geometry.TriangleMesh:
    """
    Build a thin rectangle (as a thin box) at the requested face of an object-space AABB,
    aligned to object axes (obj_axes).
    """
    k, s = _face_id_to_axis_sign(face_id)

    # face coordinate in local
    xk = mx_l[k] if s > 0 else mn_l[k]

    # rectangle spans [mn,mx] on the other two axes, and a thin thickness on k
    mn2 = mn_l.copy()
    mx2 = mx_l.copy()
    mn2[k] = xk - thickness * 0.5
    mx2[k] = xk + thickness * 0.5

    return _aabb_mesh_in_object_space(mn2, mx2, origin, obj_axes, color=color)


# ----------------------------
# Main visualization entry
# ----------------------------

def verify_relations_vis(
    pts: np.ndarray,
    assigned_ids: np.ndarray,
    bboxes_by_name: Dict[str, Any],
    symmetry: Any,
    attachments: List[Dict[str, Any]],
    object_space: Dict[str, Any],
    containment: Any,
    vis_anchor_points: bool = True,
    anchor_radius: float = 0.002,
    ignore_unknown: bool = False,
) -> None:
    """
    Visualize per-label relations.

    Requested behavior:
      - For each label: 4 views.
      - View 1: identical to before (overlay full shape + label+all-neighbors + bboxes).
      - Views 2-4: relation-specific neighbor filtering:
          - show bbox + point cloud colored ONLY for neighbors that participate in that attachment kind
            (volume / face / point) with the current label
          - do NOT show bboxes for unrelated neighbors
          - do NOT color unrelated neighbors as "neighbors"
        (We still keep the rest of the points as light gray so you keep context.)
    """
    pts = np.asarray(pts, dtype=np.float64)
    assigned_ids = np.asarray(assigned_ids).reshape(-1).astype(np.int32)
    if pts.shape[0] != assigned_ids.shape[0]:
        raise ValueError(f"Point count mismatch: pts={pts.shape[0]} ids={assigned_ids.shape[0]}")

    origin = np.array(object_space["origin"], dtype=np.float64)
    obj_axes = np.array(object_space["axes"], dtype=np.float64)
    if obj_axes.shape != (3, 3):
        raise ValueError("object_space['axes'] must be 3x3 (columns are u0,u1,u2)")

    name_to_id = _get_name_to_id(bboxes_by_name)

    # prebuild per-name OBB + lineset
    name_to_obb: Dict[str, o3d.geometry.OrientedBoundingBox] = {}
    name_to_ls: Dict[str, o3d.geometry.LineSet] = {}
    for name, entry in bboxes_by_name.items():
        try:
            obb = _obb_from_entry(entry)
            name_to_obb[name] = obb
            name_to_ls[name] = _lineset_from_obb(obb, color=(0.0, 0.0, 0.0))
        except Exception:
            continue

    all_names = sorted(list(name_to_id.keys()))
    if ignore_unknown:
        all_names = [n for n in all_names if "unknown" not in n.lower()]

    # Base point cloud (for overlay view)
    pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd_all.paint_uniform_color((0.80, 0.80, 0.80))

    def _neighbor_mask_from_names(neigh_names: List[str]) -> np.ndarray:
        mask = np.zeros((assigned_ids.shape[0],), dtype=bool)
        for nn in neigh_names:
            if nn not in name_to_id:
                continue
            nid = name_to_id[nn]
            mask |= (assigned_ids == nid)
        return mask

    def _add_bbox_lines(geoms: List[o3d.geometry.Geometry], self_name: str, neigh_names: List[str]) -> None:
        if self_name in name_to_ls:
            geoms.append(name_to_ls[self_name])
        for nn in neigh_names:
            if nn in name_to_ls:
                geoms.append(name_to_ls[nn])

    for name in all_names:
        if name not in name_to_id:
            continue

        lid = name_to_id[name]
        highlight_mask = (assigned_ids == lid)

        # neighbors from ALL attachment edges (for overlay view)
        neigh_names_all = _neighbor_names(attachments, name)
        neighbor_mask_all = _neighbor_mask_from_names(neigh_names_all)

        # ------------------------------------------
        # View 1 (overlay): label + ALL neighbors + full shape + bboxes
        # ------------------------------------------
        colors1 = _colorize_assigned_ids(assigned_ids, highlight_mask, neighbor_mask_all)
        pcd_sel1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd_sel1.colors = o3d.utility.Vector3dVector(colors1)

        geoms1: List[o3d.geometry.Geometry] = [pcd_all, pcd_sel1]

        if name in name_to_ls:
            ls = name_to_ls[name]
            ls.paint_uniform_color((0.0, 0.0, 0.0))
            geoms1.append(ls)

        for nn in neigh_names_all:
            if nn in name_to_ls:
                ls = name_to_ls[nn]
                ls.paint_uniform_color((0.1, 0.1, 0.1))
                geoms1.append(ls)

        o3d.visualization.draw_geometries(
            geoms1,
            window_name=f"[VIS 1/4] Overlay: {name} + all neighbors",
        )

        # per-label attachment edges
        att_edges = _attachments_for_label(attachments, name)

        # ---------------------------------------------------
        # View 2: volumetric attachments (only volume-neighbors)
        # ---------------------------------------------------
        neigh_names_vol = _neighbor_names_for_kind(att_edges, name, "volume")
        neighbor_mask_vol = _neighbor_mask_from_names(neigh_names_vol)

        colors2 = _colorize_assigned_ids(assigned_ids, highlight_mask, neighbor_mask_vol)
        pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd2.colors = o3d.utility.Vector3dVector(colors2)

        geoms2: List[o3d.geometry.Geometry] = [pcd2]
        _add_bbox_lines(geoms2, name, neigh_names_vol)

        # add overlap volumes only for volume-neighbors
        if name in name_to_obb:
            obb_a = name_to_obb[name]
            mnA, mxA = _aabb_minmax_in_object_from_obb(obb_a, origin, obj_axes)

            for e in att_edges:
                if _infer_attachment_kind(e) != "volume":
                    continue
                other = _other_name_in_edge(e, name)
                if other is None or other not in name_to_obb:
                    continue
                if other not in neigh_names_vol:
                    continue

                obb_b = name_to_obb[other]
                mnB, mxB = _aabb_minmax_in_object_from_obb(obb_b, origin, obj_axes)
                mnI, mxI, volI = _intersection_aabb(mnA, mxA, mnB, mxB)
                if mnI is None:
                    continue
                geoms2.append(_aabb_mesh_in_object_space(mnI, mxI, origin, obj_axes, color=(1.0, 0.0, 0.0)))

        o3d.visualization.draw_geometries(
            geoms2,
            window_name=f"[VIS 2/4] Volumetric attachments: {name} (filtered neighbors)",
        )

        # ------------------------------------------
        # View 3: face attachments (only face-neighbors)
        # ------------------------------------------
        neigh_names_face = _neighbor_names_for_kind(att_edges, name, "face")
        neighbor_mask_face = _neighbor_mask_from_names(neigh_names_face)

        colors3 = _colorize_assigned_ids(assigned_ids, highlight_mask, neighbor_mask_face)
        pcd3 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd3.colors = o3d.utility.Vector3dVector(colors3)

        geoms3: List[o3d.geometry.Geometry] = [pcd3]
        _add_bbox_lines(geoms3, name, neigh_names_face)

        # face slabs only for face-neighbors
        if name in name_to_obb:
            obb_a = name_to_obb[name]
            mnA, mxA = _aabb_minmax_in_object_from_obb(obb_a, origin, obj_axes)

            for e in att_edges:
                if _infer_attachment_kind(e) != "face":
                    continue
                other = _other_name_in_edge(e, name)
                if other is None or other not in name_to_obb:
                    continue
                if other not in neigh_names_face:
                    continue

                a_face = e.get("a_face", None)
                b_face = e.get("b_face", None)
                if not isinstance(a_face, str) or not isinstance(b_face, str):
                    continue

                # If current label is "b", swap faces so "a_face" always corresponds to current label in visualization
                if e.get("b") == name:
                    a_face, b_face = b_face, a_face

                obb_b = name_to_obb[other]
                mnB, mxB = _aabb_minmax_in_object_from_obb(obb_b, origin, obj_axes)

                geoms3.append(_quad_mesh_for_face_in_object_aabb(
                    mnA, mxA, a_face, origin, obj_axes,
                    thickness=max(1e-4, float(anchor_radius) * 0.2),
                    color=(0.0, 0.2, 1.0),
                ))
                geoms3.append(_quad_mesh_for_face_in_object_aabb(
                    mnB, mxB, b_face, origin, obj_axes,
                    thickness=max(1e-4, float(anchor_radius) * 0.2),
                    color=(0.0, 1.0, 0.2),
                ))

        o3d.visualization.draw_geometries(
            geoms3,
            window_name=f"[VIS 3/4] Face attachments: {name} (filtered neighbors)",
        )

        # ------------------------------------------
        # View 4: anchor-point attachments (only point-neighbors)
        # ------------------------------------------
        neigh_names_point = _neighbor_names_for_kind(att_edges, name, "point")
        neighbor_mask_point = _neighbor_mask_from_names(neigh_names_point)

        colors4 = _colorize_assigned_ids(assigned_ids, highlight_mask, neighbor_mask_point)
        pcd4 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd4.colors = o3d.utility.Vector3dVector(colors4)

        geoms4: List[o3d.geometry.Geometry] = [pcd4]
        _add_bbox_lines(geoms4, name, neigh_names_point)

        if vis_anchor_points:
            big_r = float(anchor_radius) * 10.0
            for e in att_edges:
                if _infer_attachment_kind(e) != "point":
                    continue
                other = _other_name_in_edge(e, name)
                if other is None:
                    continue
                if other not in neigh_names_point:
                    continue

                aw = e.get("anchor_world", None)
                if aw is None:
                    al = e.get("anchor_local", None)
                    if al is not None:
                        al = np.array(al, dtype=np.float64).reshape(3)
                        aw = _object_to_world(al, origin, obj_axes).tolist()
                if aw is None:
                    continue
                aw = np.array(aw, dtype=np.float64).reshape(3)
                geoms4.append(_sphere(aw, big_r, color=(1.0, 0.0, 0.0)))

        o3d.visualization.draw_geometries(
            geoms4,
            window_name=f"[VIS 4/4] Anchor-point attachments: {name} (filtered neighbors)",
        )


# Backward-compat alias (so launcher can always import verify_relations_vis)
def verify_relations(**kwargs):
    return verify_relations_vis(**kwargs)
