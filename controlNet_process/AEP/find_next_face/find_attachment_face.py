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


def _get_face_corners(center: np.ndarray, axes: np.ndarray, extents: np.ndarray,
                      axis: int, sign: int) -> np.ndarray:
    """Get 4 corners of a face in world coordinates."""
    other_axes = [0, 1, 2]
    other_axes.remove(axis)
    b, c = other_axes[0], other_axes[1]
    
    corners = []
    for sb in [-1.0, 1.0]:
        for sc in [-1.0, 1.0]:
            offset = np.zeros(3)
            offset[axis] = sign * extents[axis]
            offset[b] = sb * extents[b]
            offset[c] = sc * extents[c]
            
            corner = center + axes @ offset
            corners.append(corner)
    
    return np.array(corners)


def _make_face_mesh(center: np.ndarray, axes: np.ndarray, extents: np.ndarray,
                    axis: int, sign: int, color: Tuple[float, float, float]):
    """Create a mesh for visualizing a face."""
    corners = _get_face_corners(center, axes, extents, axis, sign)
    pts = np.array([corners[0], corners[1], corners[3], corners[2]])
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts)
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(list(color))
    
    return mesh


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

    # Get the edited face from the edit
    edited_face = t_face
    edited_axis, edited_sign = k_t, s_t

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

    # COMPUTE CANDIDATES based on NEW criteria
    print("\n" + "="*60)
    print("COMPUTING CANDIDATE FACES")
    print("="*60)
    print(f"Edited face (excluded): {edited_face}")
    print(f"Attached face: {attached_face}")
    
    candidates_for_vis = []
    for axis in (0, 1, 2):
        for sign in (-1, +1):
            face_name = _face_to_str(int(axis), int(sign))
            
            # CRITERIA 3: A (from BEFORE) should not be the edited face
            if int(axis) == int(edited_axis) and int(sign) == int(edited_sign):
                print(f"\nFace {face_name}: EXCLUDED (is the edited face)")
                continue
            
            # Get face centers for A (before) and A' (after)
            center_A_before = _face_center_world(neighbor_before_obb, axis, sign)
            center_A_after = _face_center_world(neighbor_after_obb, axis, sign)
            
            # Face normal in BEFORE frame
            normal_before = R0[:, axis] * float(sign)
            normal_before = _unit(normal_before)
            
            # CRITERIA 1: A should NOT be the same as A' (should be translated along normal)
            translation = center_A_after - center_A_before
            translation_magnitude = float(np.linalg.norm(translation))
            
            if translation_magnitude <= min_translation:
                print(f"\nFace {face_name}: No translation ({translation_magnitude:.6e})")
                continue
            
            # Check if translation is along normal
            translation_normalized = translation / translation_magnitude
            alignment = float(np.dot(translation_normalized, normal_before))
            is_aligned = abs(alignment) >= normal_alignment_min
            
            # CRITERIA 2: A and A' should have the same boundary size (extents)
            extent_before = float(E0[axis])
            extent_after = float(E1_raw[axis])
            extent_diff = abs(extent_after - extent_before)
            same_size = extent_diff < extents_tol
            
            # Get boundary lengths (the OTHER two dimensions)
            other_axes = [0, 1, 2]
            other_axes.remove(axis)
            b, c = other_axes[0], other_axes[1]
            
            boundary_before = (float(E0[b]), float(E0[c]))
            boundary_after = (float(E1_raw[b]), float(E1_raw[c]))
            boundary_diff = (abs(boundary_after[0] - boundary_before[0]), 
                           abs(boundary_after[1] - boundary_before[1]))
            boundary_same = all(d < extents_tol for d in boundary_diff)
            
            print(f"\nFace {face_name}:")
            print(f"  Translation magnitude: {translation_magnitude:.6f}")
            print(f"  Alignment with normal: {alignment:.6f} (>={normal_alignment_min}): {is_aligned}")
            print(f"  Extent (normal direction): before={extent_before:.6f}, after={extent_after:.6f}, diff={extent_diff:.6e}")
            print(f"  Boundary size: before={boundary_before}, after={boundary_after}, same={boundary_same}")
            print(f"  Same size: {same_size and boundary_same}")
            
            # All criteria must be satisfied
            if is_aligned and same_size and boundary_same:
                candidates_for_vis.append({
                    'axis': axis,
                    'sign': sign,
                    'face': face_name,
                    'translation_magnitude': translation_magnitude,
                    'alignment': alignment,
                })
                print(f"  ✓ CANDIDATE")
    
    print("\n" + "="*60)
    print(f"Found {len(candidates_for_vis)} candidate(s)")
    print("="*60)
    
    # Select best candidate (highest translation magnitude)
    selected_face_for_vis = None
    if candidates_for_vis:
        candidates_for_vis.sort(key=lambda x: x['translation_magnitude'], reverse=True)
        selected_face_for_vis = candidates_for_vis[0]
        print(f"\nSelected face for visualization: {selected_face_for_vis['face']}")
    
    # VIS: Show neighbor before and after with selected face
    if o3d is not None:
        geoms = []
        
        # Neighbor BEFORE (BLUE)
        obb0 = o3d.geometry.OrientedBoundingBox(center=C0, R=R0, extent=2.0 * E0)
        ls0 = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb0)
        ls0.paint_uniform_color([0.0, 0.0, 1.0])  # BLUE
        geoms.append(ls0)
        
        # Neighbor AFTER (RED)
        obb1 = o3d.geometry.OrientedBoundingBox(center=C1_raw, R=R1_raw, extent=2.0 * E1_raw)
        ls1 = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb1)
        ls1.paint_uniform_color([1.0, 0.0, 0.0])  # RED
        geoms.append(ls1)
        
        # Selected face A (YELLOW) on BEFORE
        if selected_face_for_vis:
            try:
                face_A_mesh = _make_face_mesh(C0, R0, E0, 
                                             selected_face_for_vis['axis'], 
                                             selected_face_for_vis['sign'], 
                                             (1.0, 1.0, 0.0))
                geoms.append(face_A_mesh)
            except Exception as e:
                print(f"Could not visualize face A: {e}")
        
        # Selected face A' (GREEN) on AFTER
        if selected_face_for_vis:
            try:
                face_A_prime_mesh = _make_face_mesh(C1_raw, R1_raw, E1_raw, 
                                                   selected_face_for_vis['axis'], 
                                                   selected_face_for_vis['sign'], 
                                                   (0.0, 1.0, 0.0))
                geoms.append(face_A_prime_mesh)
            except Exception as e:
                print(f"Could not visualize face A': {e}")
        
        title = f"Selected: {selected_face_for_vis['face'] if selected_face_for_vis else 'None'} | A (YELLOW on blue) -> A' (GREEN on red)"
        o3d.visualization.draw_geometries(geoms, window_name=title)

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
    p0 = _face_center_world(neighbor_before_obb, axis, sign)
    p1_fixed = _face_center_world(neighbor_after_obb, axis, sign)

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
            "after_obb": neighbor_after_obb,
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