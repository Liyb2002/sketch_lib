#!/usr/bin/env python3
# AEP/find_next_face/find_attachment_face.py
#
# ATTACHMENT logic: Find the passive face (opposite of attachment face)

from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


def _as_np(x) -> np.ndarray:
    """Convert input to numpy array."""
    return np.asarray(x, dtype=np.float64)


def _parse_face(face: str) -> Tuple[int, int]:
    """Parse face string like '+u0' into (axis, sign)."""
    if not isinstance(face, str) or len(face) != 3 or face[0] not in "+-" or face[1] != "u" or face[2] not in "012":
        raise ValueError(f"Invalid face '{face}'. Must be one of: +u0 -u0 +u1 -u1 +u2 -u2")
    s = +1 if face[0] == "+" else -1
    k = int(face[2])
    return k, s


def _face_to_str(axis: int, sign: int) -> str:
    """Convert (axis, sign) to face string."""
    s = "+" if int(sign) >= 0 else "-"
    return f"{s}u{int(axis)}"


def _axes_cols(axes_list: Any) -> np.ndarray:
    """Treat JSON axes as COLUMNS."""
    R = _as_np(axes_list)
    if R.shape != (3, 3):
        raise ValueError(f"axes must be 3x3, got {R.shape}")
    return R


def _unit(v: np.ndarray) -> np.ndarray:
    """Return unit vector."""
    v = _as_np(v).reshape(3,)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _face_center_world(obb: Dict[str, Any], axis: int, sign: int) -> np.ndarray:
    """Calculate face center in world coordinates."""
    C = _as_np(obb["center"])
    R = _axes_cols(obb["axes"])
    E = _as_np(obb["extents"])
    return C + float(sign) * float(E[axis]) * R[:, axis]


def _all_face_centers_world(obb: Dict[str, Any]) -> List[Tuple[int, int, np.ndarray]]:
    """Get all 6 face centers."""
    out = []
    for axis in (0, 1, 2):
        for sign in (-1, +1):
            out.append((axis, sign, _face_center_world(obb, axis, sign)))
    return out


def _get_target_edit_obbs_and_face(edit: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Extract OBBs and face from edit dict."""
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
    normal_alignment_min: float = 0.7,
    vis: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Find passive face for ATTACHMENT connection.
    Returns list of face_edit_change structures (without 'target' field), or None if no candidates found.
    
    Main functionality: Establish face correspondence and find faces translated along normals.
    
    Args:
        vis: If True, show 3D visualization (requires Open3D)
    """
    
    # Extract edit information
    t_before, t_after, t_face = _get_target_edit_obbs_and_face(edit)
    k_t, s_t = _parse_face(t_face)
    t_face_center0 = _face_center_world(t_before, axis=k_t, sign=s_t)

    # Find attachment face (nearest neighbor face to target's edited face)
    best_dist = 1e30
    attached_axis, attached_sign = 0, 1
    
    for axis, sign, p in _all_face_centers_world(neighbor_before_obb):
        d = float(np.linalg.norm(p - t_face_center0))
        if d < best_dist:
            best_dist = d
            attached_axis, attached_sign = axis, sign

    attached_face = _face_to_str(attached_axis, attached_sign)

    # Get neighbor OBB data
    C0 = _as_np(neighbor_before_obb["center"]).reshape(3,)
    R0 = _axes_cols(neighbor_before_obb["axes"])
    E0 = _as_np(neighbor_before_obb["extents"]).reshape(3,)
    
    C1 = _as_np(neighbor_after_obb["center"]).reshape(3,)
    R1 = _axes_cols(neighbor_after_obb["axes"])
    E1 = _as_np(neighbor_after_obb["extents"]).reshape(3,)

    # Establish face correspondence and check for translation along normals
    
    candidate_faces = []
    
    for axis in (0, 1, 2):
        for sign in (-1, +1):
            face_name = _face_to_str(axis, sign)
            
            # Get face centers A (before) and A' (after)
            center_A = _face_center_world(neighbor_before_obb, axis, sign)
            center_A_prime = _face_center_world(neighbor_after_obb, axis, sign)
            
            # Get face normal in BEFORE frame
            normal_A = R0[:, axis] * float(sign)
            normal_A_unit = _unit(normal_A)
            
            # Calculate translation vector from A to A'
            translation = center_A_prime - center_A
            translation_magnitude = float(np.linalg.norm(translation))
            
            # Check criterion: translation should be non-zero
            if translation_magnitude < min_translation:
                continue
            
            # Check if translation is along the normal
            translation_unit = translation / translation_magnitude
            alignment = float(np.dot(translation_unit, normal_A_unit))
            
            # Check if translation is along normal (alignment close to ±1)
            if abs(alignment) >= normal_alignment_min:
                candidate_faces.append({
                    'axis': axis,
                    'sign': sign,
                    'face_name': face_name,
                    'center_before': center_A,
                    'center_after': center_A_prime,
                    'normal': normal_A_unit,
                    'translation': translation,
                    'translation_magnitude': translation_magnitude,
                    'alignment': alignment,
                })
            else:
                continue

    # Visualize if requested and Open3D is available
    if vis and o3d is not None and len(candidate_faces) > 0:
        geoms = []
        
        # Neighbor BEFORE (BLUE wireframe)
        obb0 = o3d.geometry.OrientedBoundingBox(center=C0, R=R0, extent=2.0 * E0)
        ls0 = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb0)
        ls0.paint_uniform_color([0.0, 0.0, 1.0])
        geoms.append(ls0)
        
        # Neighbor AFTER (RED wireframe)
        obb1 = o3d.geometry.OrientedBoundingBox(center=C1, R=R1, extent=2.0 * E1)
        ls1 = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb1)
        ls1.paint_uniform_color([1.0, 0.0, 0.0])
        geoms.append(ls1)
        
        # Visualize only candidate faces on AFTER (A')
        colors = [
            (1.0, 1.0, 0.0),  # Yellow
            (0.0, 1.0, 1.0),  # Cyan
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 0.0),  # Green
            (1.0, 0.5, 0.0),  # Orange
            (0.5, 0.0, 1.0),  # Purple
        ]
        
        for i, candidate in enumerate(candidate_faces):
            color = colors[i % len(colors)]
            
            # Only visualize A' (face on AFTER)
            try:
                face_A_prime = _make_face_mesh(C1, R1, E1, 
                                              candidate['axis'], 
                                              candidate['sign'], 
                                              color)
                geoms.append(face_A_prime)
            except Exception as e:
                print(f"Could not visualize face A' for {candidate['face_name']}: {e}")
        
        title = f"Found {len(candidate_faces)} candidate(s) on AFTER (red) | BEFORE (blue)"
        o3d.visualization.draw_geometries(geoms, window_name=title)

    # Build results for all candidates, or return None if no candidates
    if not candidate_faces:
        return None
    
    results = []
    candidate_faces.sort(key=lambda x: x['translation_magnitude'], reverse=True)
    
    for candidate in candidate_faces:
        axis = candidate['axis']
        sign = candidate['sign']
        face_name = candidate['face_name']
        
        # Calculate delta along normal
        n_hat = candidate['normal']
        delta_face = float(np.dot(candidate['translation'], n_hat))
        
        # Get extents
        old_extent = float(E0[axis])
        new_extent = float(E1[axis])
        extent_delta = new_extent - old_extent
        
        # Calculate ratio
        ratio = abs(delta_face) / max(abs(old_extent), 1e-12)
        
        # Determine expand or shrink
        if abs(extent_delta) < extents_tol:
            expand_or_shrink = "translate"
        elif extent_delta > 0:
            expand_or_shrink = "expand"
        else:
            expand_or_shrink = "shrink"
        
        # Face centers
        p0 = candidate['center_before']
        p1 = candidate['center_after']
        
        # Center motion
        vC = C1 - C0
        vC_norm = float(np.linalg.norm(vC))
        
        result = {
            "connection_type": "attachment",
            "change": {
                "type": "move_single_face",
                "delta_ratio_min": float(ratio),
                "delta_ratio_max": float(ratio),
                "ratio_sampled": float(ratio),
                "expand_or_shrink": expand_or_shrink,
                "face": face_name,
                "axis": int(axis),
                "axis_name": f"u{int(axis)}",
                "sign": int(sign),
                "delta_requested": float(delta_face),
                "delta_applied": float(delta_face),
                "old_extent": float(old_extent),
                "new_extent": float(new_extent),
                "min_extent": 1e-4,
                "before_obb": neighbor_before_obb,
                "after_obb": neighbor_after_obb,
                "diagnostics": {
                    "selection_reason": "corresponding_face_attachment",
                    "target_edited_face": t_face,
                    "neighbor_corresponding_face": face_name,
                    "axis_mapping": f"axis_{axis}_sign_{sign}",
                    "neighbor_center_delta": vC.tolist(),
                    "neighbor_center_delta_norm": float(vC_norm),
                    "face_center_before": p0.tolist(),
                    "face_center_after": p1.tolist(),
                    "face_normal_world_before": n_hat.tolist(),
                    "extent_delta": float(extent_delta),
                    "translation_magnitude": candidate['translation_magnitude'],
                    "normal_alignment": candidate['alignment'],
                },
                "input_edit_debug": {
                    "target_edit_before_obb": t_before,
                    "target_edit_after_obb": t_after,
                    "target_edit_face": t_face,
                },
            },
        }
        results.append(result)
        
    return results