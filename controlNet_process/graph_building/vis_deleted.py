#!/usr/bin/env python3
# graph_building/vis_deleted.py

import numpy as np
import open3d as o3d
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Helpers: bbox parsing / transforms (mirrors graph_building/vis.py expectations)
# ----------------------------
def _get_obb_pca(bbox_entry: Dict[str, Any]) -> Dict[str, Any]:
    if "obb_pca" not in bbox_entry:
        raise KeyError("bbox entry missing 'obb_pca'")
    obb = bbox_entry["obb_pca"]
    if "center" not in obb or "extents" not in obb or "axes" not in obb:
        raise KeyError("obb_pca missing 'center'/'extents'/'axes'")
    return obb


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


def _world_to_object(p_world: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return (p_world - origin) @ axes


def _object_to_world(p_local: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return origin + axes @ p_local


def _sphere(center: np.ndarray, radius: float, color=(1.0, 0.0, 0.0)) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
    s.translate(center.astype(np.float64))
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    return s


def _get_name_to_id(bboxes_by_name: Dict[str, Any]) -> Dict[str, int]:
    out = {}
    for name, entry in bboxes_by_name.items():
        if isinstance(entry, dict) and "label_id" in entry:
            out[name] = int(entry["label_id"])
    return out


def _other_name_in_edge(e: Dict[str, Any], name: str) -> Optional[str]:
    a = e.get("a", None)
    b = e.get("b", None)
    if a == name and isinstance(b, str):
        return b
    if b == name and isinstance(a, str):
        return a
    return None


def _attachments_for_label(attachments: List[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    out = []
    for e in attachments:
        if e.get("a") == name or e.get("b") == name:
            out.append(e)
    return out


def _neighbor_names_from_edges(att_edges: List[Dict[str, Any]], name: str) -> List[str]:
    neigh = set()
    for e in att_edges:
        other = _other_name_in_edge(e, name)
        if other is not None:
            neigh.add(other)
    return sorted(list(neigh))


def _colorize_assigned_ids_deleted_only(
    assigned_ids: np.ndarray,
    highlight_mask: np.ndarray,
    neighbor_mask: np.ndarray,
) -> np.ndarray:
    """
    Return (N,3) colors:
      - background: light gray
      - deleted neighbors: orange
      - highlight (label): blue
      - unknown ids (<0): darker gray
    """
    N = assigned_ids.shape[0]
    colors = np.zeros((N, 3), dtype=np.float64)
    colors[:] = np.array([0.75, 0.75, 0.75], dtype=np.float64)

    colors[neighbor_mask] = np.array([1.0, 0.6, 0.1], dtype=np.float64)
    colors[highlight_mask] = np.array([0.1, 0.3, 1.0], dtype=np.float64)

    colors[assigned_ids < 0] = np.array([0.4, 0.4, 0.4], dtype=np.float64)
    return colors


def _neighbor_mask_from_names(assigned_ids: np.ndarray, name_to_id: Dict[str, int], neigh_names: List[str]) -> np.ndarray:
    mask = np.zeros((assigned_ids.shape[0],), dtype=bool)
    for nn in neigh_names:
        if nn not in name_to_id:
            continue
        nid = name_to_id[nn]
        mask |= (assigned_ids == nid)
    return mask


def _infer_attachment_kind(e: Dict[str, Any]) -> str:
    # same heuristic as vis.py, but we only need it to decide anchor display
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
    if any(x in e for x in ["overlap_volume", "vol_overlap", "overlap_box_local_min", "overlap_box_local_max"]):
        return "volume"
    if ("a_face" in e) and ("b_face" in e):
        return "face"
    return "point"


# ----------------------------
# Main entry
# ----------------------------
def verify_relations_vis_deleted(
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
    Visualize ONLY deleted attachment relations.

    For each label:
      - If it has no deleted edges -> skip (no window)
      - Otherwise, show ONE window:
          - pcd colored: label (blue), deleted-neighbors (orange), rest gray
          - bboxes: only label + deleted-neighbors
          - anchor spheres for point-type deleted edges (if present)
    """
    pts = np.asarray(pts, dtype=np.float64)
    assigned_ids = np.asarray(assigned_ids).reshape(-1).astype(np.int32)
    if pts.shape[0] != assigned_ids.shape[0]:
        raise ValueError(f"Point count mismatch: pts={pts.shape[0]} ids={assigned_ids.shape[0]}")

    origin = np.array(object_space["origin"], dtype=np.float64)
    obj_axes = np.array(object_space["axes"], dtype=np.float64)
    if obj_axes.shape != (3, 3):
        raise ValueError("object_space['axes'] must be 3x3")

    name_to_id = _get_name_to_id(bboxes_by_name)

    # build OBB linesets
    name_to_ls: Dict[str, o3d.geometry.LineSet] = {}
    for name, entry in bboxes_by_name.items():
        try:
            obb = _obb_from_entry(entry)
            name_to_ls[name] = _lineset_from_obb(obb, color=(0.0, 0.0, 0.0))
        except Exception:
            continue

    all_names = sorted(list(name_to_id.keys()))
    if ignore_unknown:
        all_names = [n for n in all_names if "unknown" not in n.lower()]

    for name in all_names:
        if name not in name_to_id:
            continue

        # only edges in deleted attachments set
        del_edges = _attachments_for_label(attachments, name)
        if len(del_edges) == 0:
            continue

        neigh_names = _neighbor_names_from_edges(del_edges, name)
        if len(neigh_names) == 0:
            continue

        lid = name_to_id[name]
        highlight_mask = (assigned_ids == lid)
        neighbor_mask = _neighbor_mask_from_names(assigned_ids, name_to_id, neigh_names)

        colors = _colorize_assigned_ids_deleted_only(assigned_ids, highlight_mask, neighbor_mask)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        geoms: List[o3d.geometry.Geometry] = [pcd]

        # bboxes: only self + deleted neighbors
        if name in name_to_ls:
            geoms.append(name_to_ls[name])
        for nn in neigh_names:
            if nn in name_to_ls:
                geoms.append(name_to_ls[nn])

        # anchor points: only for point-type deleted edges
        if vis_anchor_points:
            big_r = float(anchor_radius) * 10.0
            for e in del_edges:
                if _infer_attachment_kind(e) != "point":
                    continue
                other = _other_name_in_edge(e, name)
                if other is None:
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
                geoms.append(_sphere(aw, big_r, color=(1.0, 0.0, 0.0)))

        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"[DELETED] {name} + {len(neigh_names)} deleted neighbors",
        )


def verify_relations_deleted(**kwargs):
    return verify_relations_vis_deleted(**kwargs)
