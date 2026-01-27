#!/usr/bin/env python3
"""
Find next edit face by detecting face translations along normals
"""

from typing import Dict, Any, List, Tuple
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


def _parse_face(face: str) -> Tuple[int, int]:
    """Parse face string like '+u0', '-u1' into (axis, sign)."""
    if not isinstance(face, str) or len(face) != 3 or face[0] not in "+-" or face[1] != "u" or face[2] not in "012":
        raise ValueError(f"Invalid face '{face}'. Must be one of: +u0 -u0 +u1 -u1 +u2 -u2")
    s = +1 if face[0] == "+" else -1
    k = int(face[2])
    return k, s


def _face_to_str(axis: int, sign: int) -> str:
    """Convert axis and sign to face string like '+u0', '-u1', etc."""
    s = "+" if int(sign) >= 0 else "-"
    return f"{s}u{int(axis)}"


def _get_face_center(center: np.ndarray, axes: np.ndarray, extents: np.ndarray, 
                     axis: int, sign: int) -> np.ndarray:
    """Calculate the world position of a face center."""
    return center + float(sign) * extents[axis] * axes[:, axis]


def _get_face_corners(center: np.ndarray, axes: np.ndarray, extents: np.ndarray,
                      axis: int, sign: int) -> np.ndarray:
    """Get 4 corners of a face in world coordinates."""
    # Get the other two axes
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
    # Reorder for proper quad: [0,1,3,2]
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
    Find faces that have translated along their normals.
    Visualizes only the final selected face in GREEN on the AFTER (red) OBB.
    Excludes the original edited face A and its correspondence A'.
    """
    if o3d is None:
        print("ERROR: open3d is required for visualization")
        return {}
    
    # Extract the original edited face from the edit dict
    edited_face = None
    if edit and isinstance(edit, dict):
        change = edit.get("change", {})
        if change:
            edited_face = change.get("face")
    
    # Parse the edited face to get axis and sign
    edited_axis, edited_sign = None, None
    if edited_face:
        try:
            edited_axis, edited_sign = _parse_face(edited_face)
            print(f"\n[INFO] Original edited face: {edited_face} (axis={edited_axis}, sign={edited_sign})")
            print(f"[INFO] Will exclude this face from candidates\n")
        except Exception as e:
            print(f"[WARNING] Could not parse edited face '{edited_face}': {e}")
    
    # Extract before OBB data
    C0 = np.asarray(neighbor_before_obb["center"], dtype=np.float64)
    R0 = np.asarray(neighbor_before_obb["axes"], dtype=np.float64)
    E0 = np.asarray(neighbor_before_obb["extents"], dtype=np.float64)
    
    # Extract after OBB data
    C1 = np.asarray(neighbor_after_obb["center"], dtype=np.float64)
    R1 = np.asarray(neighbor_after_obb["axes"], dtype=np.float64)
    E1 = np.asarray(neighbor_after_obb["extents"], dtype=np.float64)
    
    print("\n" + "="*60)
    print("FACE CORRESPONDENCE AND TRANSLATION ANALYSIS")
    print("="*60)
    
    # Analyze all 6 faces
    candidates = []
    
    for axis in [0, 1, 2]:
        for sign in [-1, 1]:
            face_name = _face_to_str(axis, sign)
            
            # EXCLUSION CRITERIA: Skip the original edited face A (and its correspondence A')
            if edited_axis is not None and axis == edited_axis and sign == edited_sign:
                print(f"\nFace: {face_name} (axis={axis}, sign={sign:+d})")
                print(f"  ✗ EXCLUDED: This is the original edited face A (and A')")
                continue
            
            # Get face centers
            center_before = _get_face_center(C0, R0, E0, axis, sign)
            center_after = _get_face_center(C1, R1, E1, axis, sign)
            
            # Face normal in BEFORE frame
            normal_before = R0[:, axis] * float(sign)
            normal_before = normal_before / np.linalg.norm(normal_before)
            
            # Translation vector
            translation = center_after - center_before
            translation_magnitude = np.linalg.norm(translation)
            
            # Check if translation is along normal
            if translation_magnitude > min_translation:
                translation_normalized = translation / translation_magnitude
                alignment = float(np.dot(translation_normalized, normal_before))
                
                # Check if aligned with normal (positive or negative direction)
                is_aligned = abs(alignment) >= normal_alignment_min
                
                print(f"\nFace: {face_name} (axis={axis}, sign={sign:+d})")
                print(f"  Center before: {center_before}")
                print(f"  Center after:  {center_after}")
                print(f"  Translation:   {translation}")
                print(f"  Translation magnitude: {translation_magnitude:.6f}")
                print(f"  Normal before: {normal_before}")
                print(f"  Alignment with normal: {alignment:.6f}")
                print(f"  Is aligned (>={normal_alignment_min}): {is_aligned}")
                
                if is_aligned:
                    candidates.append({
                        'face': face_name,
                        'axis': axis,
                        'sign': sign,
                        'center_before': center_before,
                        'center_after': center_after,
                        'translation': translation,
                        'translation_magnitude': translation_magnitude,
                        'alignment': alignment,
                    })
                    print(f"  ✓ CANDIDATE for next edit face!")
            else:
                print(f"\nFace: {face_name} - No significant translation ({translation_magnitude:.6e})")
    
    print("\n" + "="*60)
    print(f"FOUND {len(candidates)} CANDIDATE FACE(S)")
    print("="*60)
    
    # Select the best candidate (highest translation magnitude)
    selected = None
    if candidates:
        # Sort by translation magnitude and pick the largest
        candidates.sort(key=lambda x: x['translation_magnitude'], reverse=True)
        selected = candidates[0]
        print(f"\n[SELECTED] Face: {selected['face']}")
        print(f"           Translation magnitude: {selected['translation_magnitude']:.6f}")
        print(f"           Alignment: {selected['alignment']:.6f}")
    
    # Visualization
    geoms = []
    
    # Before OBB (BLUE wireframe)
    obb0 = o3d.geometry.OrientedBoundingBox(center=C0, R=R0, extent=2.0 * E0)
    ls0 = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb0)
    ls0.paint_uniform_color([0.0, 0.0, 1.0])
    geoms.append(ls0)
    
    # After OBB (RED wireframe)
    obb1 = o3d.geometry.OrientedBoundingBox(center=C1, R=R1, extent=2.0 * E1)
    ls1 = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb1)
    ls1.paint_uniform_color([1.0, 0.0, 0.0])
    geoms.append(ls1)
    
    # Visualize ONLY the selected face in GREEN (on the AFTER OBB)
    if selected:
        mesh = _make_face_mesh(C1, R1, E1, selected['axis'], selected['sign'], (0.0, 1.0, 0.0))
        geoms.append(mesh)
        window_title = f"Selected face: {selected['face']} (GREEN) | BEFORE (blue) AFTER (red)"
    else:
        window_title = "No face selected | BEFORE (blue) AFTER (red)"
    
    o3d.visualization.draw_geometries(geoms, window_name=window_title)
    
    # Return empty dict for compatibility
    return {}