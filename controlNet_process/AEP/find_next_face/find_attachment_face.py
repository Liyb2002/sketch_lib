#!/usr/bin/env python3
# AEP/find_next_face/find_attachment_face.py
#
# ATTACHMENT logic: Find the passive face (opposite of attachment face)

from __future__ import annotations
from typing import Any, Dict, Tuple, List
import numpy as np
import copy

try:
    import open3d as o3d
except Exception:
    o3d = None


def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _parse_face(face: str) -> Tuple[int, int]:
    if not isinstance(face, str) or len(face) != 3 or face[0] not in "+-" or face[1] != "u" or face[2] not in "012":
        raise ValueError(f"Invalid face '{face}'. Must be one of: +u0 -u0 +u1 -u1 +u2 -u2")
    s = +1 if face[0] == "+" else -1
    k = int(face[2])
    return k, s


def _face_to_str(axis: int, sign: int) -> str:
    s = "+" if int(sign) >= 0 else "-"
    return f"{s}u{int(axis)}"


def _axes_cols(axes_list: Any) -> np.ndarray:
    """Treat JSON axes as COLUMNS."""
    R = _as_np(axes_list)
    if R.shape != (3, 3):
        raise ValueError(f"axes must be 3x3, got {R.shape}")
    return R


def _unit(v: np.ndarray) -> np.ndarray:
    v = _as_np(v).reshape(3,)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _face_center_world(obb: Dict[str, Any], axis: int, sign: int) -> np.ndarray:
    C = _as_np(obb["center"])
    R = _axes_cols(obb["axes"])
    E = _as_np(obb["extents"])
    return C + float(sign) * float(E[axis]) * R[:, axis]


def _all_face_centers_world(obb: Dict[str, Any]) -> List[Tuple[int, int, np.ndarray]]:
    out = []
    for axis in (0, 1, 2):
        for sign in (-1, +1):
            out.append((axis, sign, _face_center_world(obb, axis, sign)))
    return out


def _axes_close(R0: np.ndarray, R1: np.ndarray, tol: float) -> Tuple[bool, float]:
    d = float(np.linalg.norm(_as_np(R0) - _as_np(R1)))
    return (d <= tol), d


def _extents_close(E0: np.ndarray, E1: np.ndarray, tol: float) -> Tuple[bool, np.ndarray]:
    E0 = _as_np(E0).reshape(3,)
    E1 = _as_np(E1).reshape(3,)
    d = E1 - E0
    ok = bool(np.all(np.abs(d) <= tol))
    return ok, d


def _translated_obb_along(obb_before: Dict[str, Any], delta: float, n_hat: np.ndarray) -> Dict[str, Any]:
    out = copy.deepcopy(obb_before)
    C0 = _as_np(obb_before["center"]).reshape(3,)
    out["center"] = (C0 + float(delta) * _unit(n_hat)).tolist()
    return out


def _make_lineset_from_obb(obb: Dict[str, Any]) -> "o3d.geometry.LineSet":
    if o3d is None:
        raise RuntimeError("open3d is required")
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
        raise RuntimeError("open3d is required")

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


def _visualize_attachment_debug(
    t_before: Dict[str, Any],
    t_after: Dict[str, Any],
    t_face: str,
    neighbor_before: Dict[str, Any],
    neighbor_after_raw: Dict[str, Any],
    neighbor_after_fixed: Dict[str, Any],
    attached_face: str,
    chosen_face: str,
    neighbor_name: str,
):
    """Visualize attachment logic for debugging"""
    if o3d is None:
        return
    
    k_t, s_t = _parse_face(t_face)
    k_att, s_att = _parse_face(attached_face)
    k_chosen, s_chosen = _parse_face(chosen_face)
    
    geoms = []
    
    # Target BEFORE (BLUE)
    tls_before = _make_lineset_from_obb(t_before)
    _color_geom(tls_before, (0.0, 0.0, 1.0))
    geoms.append(tls_before)
    
    # Target AFTER (RED)
    tls_after = _make_lineset_from_obb(t_after)
    _color_geom(tls_after, (1.0, 0.0, 0.0))
    geoms.append(tls_after)
    
    # Target's edited face (CYAN)
    try:
        t_face_mesh, t_face_border = _make_face_patch(t_after, axis=k_t, sign=s_t, color_rgb=(0.0, 1.0, 1.0))
        geoms.extend([t_face_mesh, t_face_border])
    except Exception:
        pass
    
    # Neighbor BEFORE (BLUE)
    nls_before = _make_lineset_from_obb(neighbor_before)
    _color_geom(nls_before, (0.0, 0.0, 1.0))
    geoms.append(nls_before)
    
    # Neighbor AFTER RAW (MAGENTA)
    nls_after_raw = _make_lineset_from_obb(neighbor_after_raw)
    _color_geom(nls_after_raw, (1.0, 0.0, 1.0))
    geoms.append(nls_after_raw)
    
    # Neighbor AFTER FIXED (RED)
    nls_after_fixed = _make_lineset_from_obb(neighbor_after_fixed)
    _color_geom(nls_after_fixed, (1.0, 0.0, 0.0))
    geoms.append(nls_after_fixed)
    
    # Attachment face on neighbor BEFORE (YELLOW)
    try:
        att_mesh, att_border = _make_face_patch(neighbor_before, axis=k_att, sign=s_att, color_rgb=(1.0, 1.0, 0.0))
        geoms.extend([att_mesh, att_border])
    except Exception:
        pass
    
    # Chosen passive face on neighbor AFTER FIXED (GREEN)
    try:
        chosen_mesh, chosen_border = _make_face_patch(neighbor_after_fixed, axis=k_chosen, sign=s_chosen, color_rgb=(0.0, 1.0, 0.0))
        geoms.extend([chosen_mesh, chosen_border])
    except Exception:
        pass
    
    title = f"ATTACHMENT DEBUG | neighbor={neighbor_name} | attached={attached_face} chosen={chosen_face}"
    try:
        o3d.visualization.draw_geometries(geoms, window_name=title)
    except TypeError:
        o3d.visualization.draw_geometries(geoms)


def _translated_obb_along(obb_before: Dict[str, Any], delta: float, n_hat: np.ndarray) -> Dict[str, Any]:
    out = copy.deepcopy(obb_before)
    C0 = _as_np(obb_before["center"]).reshape(3,)
    out["center"] = (C0 + float(delta) * _unit(n_hat)).tolist()
    return out


def _get_target_edit_obbs_and_face(edit: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    target = edit.get("target", None)
    if not isinstance(target, str) or not target:
        raise ValueError("edit missing valid 'target'")

    ch = edit.get("change", {}) or {}
    face = ch.get("face", None)
    if not isinstance(face, str):
        raise ValueError("edit['change'] missing 'face' string")

    before_obb = ch.get("before_obb", None)
    after_obb = ch.get("after_obb", None)
    if not isinstance(before_obb, dict) or not isinstance(after_obb, dict):
        raise ValueError("edit['change'] missing before_obb/after_obb")

    return before_obb, after_obb, face


def find_attachment_face(
    edit: Dict[str, Any],
    neighbor_before_obb: Dict[str, Any],
    neighbor_after_obb: Dict[str, Any],
    *,
    extents_tol: float = 1e-6,
    axes_tol: float = 1e-6,
    min_translation: float = 1e-8,
    normal_alignment_min: float = 0.95,
) -> Dict[str, Any]:
    """
    Find passive face for ATTACHMENT connection.
    Returns face_edit_change structure (without 'target' field).
    """
    
    # VIS: Show neighbor before and after
    if o3d is not None:
        geoms = []
        
        # Neighbor BEFORE (BLUE)
        C0 = _as_np(neighbor_before_obb["center"])
        R0 = _axes_cols(neighbor_before_obb["axes"])
        E0 = _as_np(neighbor_before_obb["extents"])
        obb0 = o3d.geometry.OrientedBoundingBox(center=C0, R=R0, extent=2.0 * E0)
        ls0 = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb0)
        ls0.paint_uniform_color([0.0, 0.0, 1.0])  # BLUE
        geoms.append(ls0)
        
        # Neighbor AFTER (RED)
        C1 = _as_np(neighbor_after_obb["center"])
        R1 = _axes_cols(neighbor_after_obb["axes"])
        E1 = _as_np(neighbor_after_obb["extents"])
        obb1 = o3d.geometry.OrientedBoundingBox(center=C1, R=R1, extent=2.0 * E1)
        ls1 = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb1)
        ls1.paint_uniform_color([1.0, 0.0, 0.0])  # RED
        geoms.append(ls1)
        
        o3d.visualization.draw_geometries(geoms, window_name="Neighbor: BEFORE (blue) AFTER (red)")
    
    t_before, t_after, t_face = _get_target_edit_obbs_and_face(edit)
    k_t, s_t = _parse_face(t_face)

    t_face_center0 = _face_center_world(t_before, axis=k_t, sign=s_t)

    # Find attachment face = nearest neighbor face center (BEFORE)
    best = None
    best_dist = 1e30
    for axis, sign, p in _all_face_centers_world(neighbor_before_obb):
        d = float(np.linalg.norm(p - t_face_center0))
        if d < best_dist:
            best_dist = d
            best = (axis, sign, p)
    if best is None:
        raise RuntimeError("Failed to infer attachment face")

    attached_axis, attached_sign, _ = best
    attached_face = _face_to_str(int(attached_axis), int(attached_sign))

    # Raw neighbor motion
    C0 = _as_np(neighbor_before_obb["center"]).reshape(3,)
    C1_raw = _as_np(neighbor_after_obb["center"]).reshape(3,)
    vC = (C1_raw - C0)
    vC_norm = float(np.linalg.norm(vC))

    # Translation-only checks (raw)
    R0 = _axes_cols(neighbor_before_obb["axes"])
    R1_raw = _axes_cols(neighbor_after_obb["axes"])
    E0 = _as_np(neighbor_before_obb["extents"]).reshape(3,)
    E1_raw = _as_np(neighbor_after_obb["extents"]).reshape(3,)

    ext_ok, ext_delta = _extents_close(E0, E1_raw, tol=extents_tol)
    axes_ok, axes_diff = _axes_close(R0, R1_raw, tol=axes_tol)

    # Candidates (exclude attachment face)
    candidates = []
    for axis in (0, 1, 2):
        for sign in (-1, +1):
            if int(axis) == int(attached_axis) and int(sign) == int(attached_sign):
                continue
            n_hat = _unit(R0[:, int(axis)] * float(sign))
            delta = float(np.dot(vC, n_hat))
            align = float(abs(delta) / max(vC_norm, 1e-12)) if vC_norm > 1e-12 else 0.0
            tang = float(np.linalg.norm(vC - delta * n_hat))
            candidates.append({
                "axis": int(axis),
                "sign": int(sign),
                "face": _face_to_str(int(axis), int(sign)),
                "n_hat": n_hat,
                "delta": delta,
                "alignment": align,
                "tangential": tang,
            })

    valid = []
    for c in candidates:
        if vC_norm < min_translation:
            continue
        if c["alignment"] < float(normal_alignment_min):
            continue
        if not ext_ok or not axes_ok:
            continue
        valid.append(c)

    if valid:
        valid.sort(key=lambda x: abs(float(x["delta"])), reverse=True)
        chosen = valid[0]
        chosen_reason = "translation_only_valid"
    else:
        # Fallback: opposite-of-attachment
        passive_axis = int(attached_axis)
        passive_sign = int(-attached_sign)
        n_hat = _unit(R0[:, passive_axis] * float(passive_sign))
        delta = float(np.dot(vC, n_hat))
        chosen = {
            "axis": passive_axis,
            "sign": passive_sign,
            "face": _face_to_str(passive_axis, passive_sign),
            "n_hat": n_hat,
            "delta": delta,
            "alignment": float(abs(delta) / max(vC_norm, 1e-12)) if vC_norm > 1e-12 else 0.0,
            "tangential": float(np.linalg.norm(vC - delta * n_hat)),
        }
        chosen_reason = "fallback_opposite_of_attachment"

    axis = int(chosen["axis"])
    sign = int(chosen["sign"])
    face = str(chosen["face"])
    n_hat = _as_np(chosen["n_hat"]).reshape(3,)
    delta_face = float(chosen["delta"])

    # Fixed translation-only after
    neighbor_after_obb_fixed = _translated_obb_along(neighbor_before_obb, delta=delta_face, n_hat=n_hat)

    # Diagnostics
    p0 = _face_center_world(neighbor_before_obb, axis, sign)
    p1_fixed = _face_center_world(neighbor_after_obb_fixed, axis, sign)

    old_extent = float(E0[axis])
    ratio = abs(delta_face) / max(abs(old_extent), 1e-12)

    # DEBUG: Print info
    print(f"\n[ATTACHMENT RESULT]")
    print(f"  Attached face: {attached_face} (dist to target face={best_dist:.6f})")
    print(f"  Chosen passive face: {face}")
    print(f"  Delta: {delta_face:.6f}")

    return {
        "connection_type": "attachment",
        "change": {
            "type": "move_single_face",
            "delta_ratio_min": float(ratio),
            "delta_ratio_max": float(ratio),
            "ratio_sampled": float(ratio),
            "expand_or_shrink": "translate",
            "face": face,
            "axis": int(axis),
            "axis_name": f"u{int(axis)}",
            "sign": int(sign),
            "delta_requested": float(delta_face),
            "delta_applied": float(delta_face),
            "old_extent": float(old_extent),
            "new_extent": float(old_extent),
            "min_extent": 1e-4,
            "before_obb": neighbor_before_obb,
            "after_obb": neighbor_after_obb_fixed,
            "diagnostics": {
                "selection_reason": str(chosen_reason),
                "attached_face": attached_face,
                "attached_face_dist_to_target_edited_face_center": float(best_dist),
                "target_edited_face": t_face,
                "target_edited_face_center_before": t_face_center0.tolist(),
                "chosen_face_alignment": float(chosen.get("alignment", 0.0)),
                "chosen_face_tangential_component": float(chosen.get("tangential", 0.0)),
                "neighbor_center_delta_raw": vC.tolist(),
                "neighbor_center_delta_norm_raw": float(vC_norm),
                "passive_face_center_before": p0.tolist(),
                "passive_face_center_after_fixed": p1_fixed.tolist(),
                "passive_face_normal_world_before": _unit(n_hat).tolist(),
                "raw_geometry_checks": {
                    "extents_tol": float(extents_tol),
                    "axes_tol": float(axes_tol),
                    "extents_unchanged_raw": bool(ext_ok),
                    "extents_delta_raw": ext_delta.tolist(),
                    "axes_unchanged_raw": bool(axes_ok),
                    "axes_diff_fro_raw": float(axes_diff),
                },
                "neighbor_after_obb_raw": neighbor_after_obb,
            },
            "input_edit_debug": {
                "target_edit_before_obb": t_before,
                "target_edit_after_obb": t_after,
                "target_edit_face": t_face,
            },
        },
    }