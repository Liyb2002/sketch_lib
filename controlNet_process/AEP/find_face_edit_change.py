#!/usr/bin/env python3
# AEP/find_face_edit_change.py
#
# Given a neighbor's before/after OBB, infer an "edited face" that represents
# a TRANSLATION along one of the OBB's local axes (face normal), while extents
# remain unchanged.
#
# Outputs:
#  - writes {counter}_face_edit_change.json into sketch/AEP/
#  - debug visualization: before OBB (blue), after OBB (red),
#    and highlight the inferred face in green.
#
# Assumptions:
#  - OBB JSON has fields: center (3), axes (3x3), extents (3)
#  - axes are treated as COLUMNS (same convention as AEP/sym_and_containment.py)
#
# No __main__ section by default (importable utility).

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _face_to_str(axis: int, sign: int) -> str:
    s = "+" if int(sign) >= 0 else "-"
    return f"{s}u{int(axis)}"


def _axes_cols(axes_list: Any) -> np.ndarray:
    """
    Treat JSON axes as columns: R is 3x3, columns are u0,u1,u2 in world coords.
    """
    R = _as_np(axes_list)
    if R.shape != (3, 3):
        raise ValueError(f"axes must be 3x3, got {R.shape}")
    return R


def _obb_face_corners_world(obb: Dict[str, Any], axis: int, sign: int) -> np.ndarray:
    """
    Return 4 corners of the specified face in world coords: (4,3).
    Face is at local coordinate q[axis] = sign * E[axis], with other axes in +/- E.
    """
    C = _as_np(obb["center"])
    R = _axes_cols(obb["axes"])     # columns
    E = _as_np(obb["extents"])

    a = int(axis)
    s = +1 if int(sign) >= 0 else -1

    # Build 4 corners in local coordinates
    # For the two remaining axes b,c:
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

    Q = np.stack(q_list, axis=0)  # (4,3)
    # world: C + R @ q  (R columns)
    P = C[None, :] + (Q @ R.T)    # since (R @ q) == (q @ R^T)
    return P


def infer_translated_face_edit(
    target: str,
    before_obb: Dict[str, Any],
    after_obb: Dict[str, Any],
    eps_extent: float = 1e-6,
    eps_axis_ortho: float = 1e-6,
) -> Dict[str, Any]:
    """
    Infer which face normal best matches the translation direction, under the constraint
    that extents are (approximately) unchanged. This is your "edited face" definition:
      - faces translated along their normal
      - size not changed (extents unchanged)

    Returns the face_edit_change dict (same schema style as your example),
    but with old_extent == new_extent when it is a pure translation.

    If extents changed significantly, we still compute a best axis/sign, but this may
    not qualify as "pure translated face" in your definition; we keep a flag.
    """
    C0 = _as_np(before_obb["center"])
    C1 = _as_np(after_obb["center"])
    dC = C1 - C0

    R0 = _axes_cols(before_obb["axes"])
    R1 = _axes_cols(after_obb["axes"])
    E0 = _as_np(before_obb["extents"])
    E1 = _as_np(after_obb["extents"])

    # Check "size not changed"
    extent_delta = E1 - E0
    extent_unchanged = bool(np.all(np.abs(extent_delta) <= float(eps_extent)))

    # Check axes are similar (optional diagnostic; not enforced)
    # If axes differ a lot, translation direction inference may be unstable.
    # We'll still proceed by using BEFORE axes as reference.
    axis_sim = float(np.max(np.abs(R1 - R0)))

    # If dC is tiny, still produce something deterministic:
    # pick axis 0, sign +, delta 0.
    if float(np.linalg.norm(dC)) < 1e-12:
        axis = 0
        sign = +1
        proj = 0.0
    else:
        # Project translation onto each local axis (world direction = R0[:,k])
        projs = np.array([float(np.dot(dC, R0[:, k])) for k in range(3)], dtype=np.float64)
        axis = int(np.argmax(np.abs(projs)))
        proj = float(projs[axis])
        sign = +1 if proj >= 0 else -1

    face = _face_to_str(axis, sign)
    old_extent = float(E0[axis])
    new_extent = float(E1[axis])

    # For a rigid translation, a natural "face translation amount" along that normal is:
    #   delta_face = proj  (because the face plane position shifts by proj along that axis normal)
    # However your existing move_single_face schema uses delta as "single-face move" that also changes extent.
    # Here we store delta_requested/applied as the *face translation* along that normal (proj).
    delta_requested = float(proj)
    delta_applied = float(proj)

    # Provide ratio fields consistent with your schema (delta / extent magnitude),
    # but clamp if extent is tiny.
    denom = max(abs(old_extent), 1e-12)
    ratio = abs(delta_applied) / denom

    # Expand/shrink isn’t semantically perfect for translation; keep a stable value:
    expand_or_shrink = "translate"

    return {
        "target": target,
        "change": {
            "type": "move_single_face",
            "delta_ratio_min": float(ratio),
            "delta_ratio_max": float(ratio),
            "ratio_sampled": float(ratio),
            "expand_or_shrink": expand_or_shrink,
            "face": face,
            "axis": int(axis),
            "axis_name": f"u{int(axis)}",
            "sign": int(sign),
            "delta_requested": float(delta_requested),
            "delta_applied": float(delta_applied),
            "old_extent": float(old_extent),
            "new_extent": float(new_extent),
            "min_extent": 1e-4,
            "before_obb": before_obb,
            "after_obb": after_obb,
            "diagnostics": {
                "extent_unchanged": bool(extent_unchanged),
                "extent_delta": extent_delta.tolist(),
                "axis_similarity_maxabs": float(axis_sim),
                "dC_world": dC.tolist(),
                "proj_on_axis": float(proj),
            },
        },
    }


def _make_lineset_from_obb(obb: Dict[str, Any]) -> "o3d.geometry.LineSet":
    """
    Create a wireframe LineSet for an oriented bounding box.
    Uses Open3D's OrientedBoundingBox for convenience.
    """
    if o3d is None:
        raise RuntimeError("open3d is required for visualization but is not available.")

    C = _as_np(obb["center"])
    R = _axes_cols(obb["axes"])  # columns
    E = _as_np(obb["extents"])

    # Open3D expects rotation matrix as 3x3 with columns as basis too.
    obb_o3d = o3d.geometry.OrientedBoundingBox(center=C, R=R, extent=2.0 * E)
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb_o3d)

    # Ensure all edges present (Open3D sometimes omits a couple in older versions; harmless)
    return ls


def _color_geom(geom, rgb: Tuple[float, float, float]):
    if hasattr(geom, "paint_uniform_color"):
        geom.paint_uniform_color(list(rgb))
    return geom


def _make_face_patch(
    obb: Dict[str, Any],
    axis: int,
    sign: int,
    color_rgb: Tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> Tuple["o3d.geometry.TriangleMesh", "o3d.geometry.LineSet"]:
    """
    Make a quad face patch (two triangles) + border line for highlighting.
    """
    if o3d is None:
        raise RuntimeError("open3d is required for visualization but is not available.")

    P = _obb_face_corners_world(obb, axis=axis, sign=sign)  # (4,3)

    # Order corners into a consistent quad:
    # P came from sb/sc loop; any consistent triangulation is fine.
    # We'll use indices [0,1,3,2] to form a non-crossing quad under typical ordering.
    pts = np.array([P[0], P[1], P[3], P[2]], dtype=np.float64)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts)
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    mesh.compute_vertex_normals()
    _color_geom(mesh, color_rgb)

    # Border
    border = o3d.geometry.LineSet()
    border.points = o3d.utility.Vector3dVector(pts)
    border.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32))
    _color_geom(border, color_rgb)

    return mesh, border


def save_face_edit_change_json(
    aep_dir: str,
    counter: int,
    face_edit_change: Dict[str, Any],
) -> str:
    os.makedirs(aep_dir, exist_ok=True)
    out_path = os.path.join(aep_dir, f"{int(counter)}_face_edit_change.json")
    with open(out_path, "w") as f:
        json.dump(face_edit_change, f, indent=2)
    return out_path


def vis_face_edit_change(
    face_edit_change: Dict[str, Any],
    overlay_ply_path: Optional[str] = None,
    show_overlay: bool = True,
    window_name: Optional[str] = None,
):
    """
    Debug visualization:
      - before OBB (blue)
      - after OBB (red)
      - highlight inferred face (green) on AFTER obb
    """
    if o3d is None:
        return  # silently skip if open3d isn't available

    ch = face_edit_change.get("change", {}) or {}
    before_obb = ch.get("before_obb", None)
    after_obb = ch.get("after_obb", None)
    face = ch.get("face", None)
    axis = int(ch.get("axis", 0))
    sign = int(ch.get("sign", +1))

    if not isinstance(before_obb, dict) or not isinstance(after_obb, dict) or not isinstance(face, str):
        return

    geoms = []

    # overlay
    if show_overlay and overlay_ply_path and os.path.isfile(overlay_ply_path):
        try:
            ply = o3d.io.read_point_cloud(overlay_ply_path)
            geoms.append(ply)
        except Exception:
            pass

    # before/after wireframes
    ls0 = _make_lineset_from_obb(before_obb)
    ls1 = _make_lineset_from_obb(after_obb)
    _color_geom(ls0, (0.0, 0.0, 1.0))  # blue
    _color_geom(ls1, (1.0, 0.0, 0.0))  # red
    geoms.extend([ls0, ls1])

    # highlight face on AFTER obb
    try:
        face_mesh, face_border = _make_face_patch(after_obb, axis=axis, sign=sign, color_rgb=(0.0, 1.0, 0.0))
        geoms.append(face_mesh)
        geoms.append(face_border)
    except Exception:
        pass

    # Draw
    title = window_name or f"face_edit_change | target={face_edit_change.get('target')} face={face}"
    try:
        o3d.visualization.draw_geometries(geoms, window_name=title)
    except TypeError:
        # older open3d versions don't support window_name
        o3d.visualization.draw_geometries(geoms)


def find_face_edit_change_and_save_and_vis(
    aep_dir: str,
    counter: int,
    target: str,
    before_obb: Dict[str, Any],
    after_obb: Dict[str, Any],
    overlay_ply_path: Optional[str] = None,
    do_vis: bool = True,
) -> str:
    """
    One-stop helper:
      - infer translated-face edit
      - save json in aep_dir as {counter}_face_edit_change.json
      - visualize (optional)
    Returns path to saved json.
    """
    face_edit_change = infer_translated_face_edit(
        target=target,
        before_obb=before_obb,
        after_obb=after_obb,
    )
    out_path = save_face_edit_change_json(aep_dir=aep_dir, counter=counter, face_edit_change=face_edit_change)

    if do_vis:
        vis_face_edit_change(
            face_edit_change,
            overlay_ply_path=overlay_ply_path,
            show_overlay=True,
            window_name=f"FaceEditChange#{counter} | {target}",
        )

    return out_path
