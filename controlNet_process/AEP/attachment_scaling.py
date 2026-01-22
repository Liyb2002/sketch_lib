# AEP/attachment_scaling.py
#
# Goal (per your corrected definition):
#   "face being edit" = the BLUE(before) OBB face that becomes BLUE-only (not overlapped by RED(after)),
#                       i.e., the face plane that MOVED the most between before->after.
#
# We will:
#   1) reconstruct target_before_obb / target_after_obb from edit_decomp in the call stack
#      (so attachment.py stays unchanged)
#   2) infer the edited face (axis, sign) from before/after geometry (NOT from attachment face)
#   3) use that inferred face outward normal to choose which NEIGHBOR face to edit
#   4) open an Open3D vis showing:
#        - target before (blue lines)
#        - target after  (red lines)
#        - inferred edited face on target-before (blue solid)
#        - same face on target-after (red solid, to show where it moved)
#        - neighbor obb (green lines)
#        - neighbor face to edit (yellow solid)
#        - neighbor opposite face (gray outline) for reference
#
# Scaling itself remains a NO-OP for now; we only find + visualize.
# No __main__ section.

from typing import Dict, Any, Tuple, List, Optional
import inspect
import numpy as np
import open3d as o3d


# ----------------------------
# Basic helpers
# ----------------------------

def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _axes_rows(axes_list) -> np.ndarray:
    # same convention as your attachment.py: axes are stored as ROWS (u0,u1,u2)
    return _as_np(axes_list)


def _face_to_str(axis: int, sign: int) -> str:
    s = "+" if int(sign) >= 0 else "-"
    return f"{s}u{int(axis)}"


def _parse_face_str(face: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(face, str):
        return None
    s = face.strip().lower().replace(" ", "")
    if not s:
        return None

    sign = +1
    if s[0] == "+":
        sign = +1
        s2 = s[1:]
    elif s[0] == "-":
        sign = -1
        s2 = s[1:]
    else:
        sign = +1
        s2 = s

    if s2.startswith("u") and len(s2) >= 2 and s2[1].isdigit():
        axis = int(s2[1])
        if axis in (0, 1, 2):
            return axis, sign

    if s2.isdigit():
        axis = int(s2)
        if axis in (0, 1, 2):
            return axis, sign

    return None


# ----------------------------
# Stack inspection (no attachment.py changes)
# ----------------------------

def _find_edit_decomp_in_stack(max_depth: int = 12) -> Optional[Dict[str, Any]]:
    """
    Find edit decomposition dict in caller stack.
    We accept local names 'edit_decomp' or 'ed'.
    """
    frames = inspect.stack()
    try:
        for fr in frames[:max_depth]:
            loc = fr.frame.f_locals
            for key in ("edit_decomp", "ed"):
                ed = loc.get(key, None)
                if isinstance(ed, dict) and all(k in ed for k in ("C0", "U0", "E0", "C1", "U1", "E1")):
                    return ed
        return None
    finally:
        del frames


def _reconstruct_target_obbs(ed: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t0 = {
        "center": _as_np(ed["C0"]).tolist(),
        "axes": _as_np(ed["U0"]).tolist(),     # rows
        "extents": _as_np(ed["E0"]).tolist(),  # half-lengths
    }
    t1 = {
        "center": _as_np(ed["C1"]).tolist(),
        "axes": _as_np(ed["U1"]).tolist(),     # rows
        "extents": _as_np(ed["E1"]).tolist(),  # half-lengths
    }
    return t0, t1


# ----------------------------
# Core: infer "edited face" from before/after geometry
# ----------------------------

def infer_edited_face_from_before_after(
    target_before_obb: Dict[str, Any],
    target_after_obb: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Your definition:
      "face being edit" = the BLUE(before) face that is not overlapped by RED(after),
      i.e. the face plane that moved the most between before -> after.

    Assumption (kept simple):
      - axes are consistent (same orientation) between before/after edit (or close enough).
      - extents are half-lengths, axes are rows.

    Compute, for each axis i and sign s in {+1,-1}, the scalar plane offset:
      plane(i,s) = dot(u_i, C) + s * E_i
    Compare before vs after:
      delta(i,s) = plane_after(i,s) - plane_before(i,s)
    Choose (i,s) with largest |delta|.
    That face is the edited face you care about.

    Returns:
      {
        "axis": int,
        "sign": int,
        "normal_world": [3],
        "delta_plane": float,
        "deltas": [{"axis":i,"sign":s,"delta":..., "abs":...}, ...] sorted desc by abs
      }
    """
    C0 = _as_np(target_before_obb["center"])
    U0 = _axes_rows(target_before_obb["axes"])
    E0 = _as_np(target_before_obb["extents"])

    C1 = _as_np(target_after_obb["center"])
    U1 = _axes_rows(target_after_obb["axes"])
    E1 = _as_np(target_after_obb["extents"])

    # Use BEFORE axes for defining the 6 faces (blue faces).
    # If U1 is slightly different, this is still usually fine for edits that keep axes.
    deltas: List[Dict[str, Any]] = []
    for i in (0, 1, 2):
        u = _normalize(U0[i])

        # plane offsets along that axis direction (scalar in world)
        p0_plus = float(np.dot(u, C0) + float(E0[i]))
        p0_minus = float(np.dot(u, C0) - float(E0[i]))

        # after plane offsets projected onto same u
        p1_plus = float(np.dot(u, C1) + float(E1[i]))
        p1_minus = float(np.dot(u, C1) - float(E1[i]))

        d_plus = p1_plus - p0_plus
        d_minus = p1_minus - p0_minus

        deltas.append({"axis": i, "sign": +1, "delta": float(d_plus), "abs": float(abs(d_plus))})
        deltas.append({"axis": i, "sign": -1, "delta": float(d_minus), "abs": float(abs(d_minus))})

    deltas.sort(key=lambda x: x["abs"], reverse=True)
    best = deltas[0]
    axis = int(best["axis"])
    sign = int(best["sign"])

    n_world = _normalize(sign * _normalize(U0[axis]))

    return {
        "axis": axis,
        "sign": sign,
        "normal_world": n_world.tolist(),
        "delta_plane": float(best["delta"]),
        "deltas": deltas,
        "note": "chosen face = max(|plane_after - plane_before|) over 6 blue faces",
    }


# ----------------------------
# Neighbor face selection (based on inferred normal)
# ----------------------------

def choose_neighbor_face_to_change(
    neighbor_obb: Dict[str, Any],
    edit_face_normal_world: np.ndarray,
) -> Dict[str, Any]:
    """
    Choose neighbor face to edit by aligning outward normals with the target edited face normal.

    Neighbor outward normals are ±u0, ±u1, ±u2 (axes are rows).
    Choose (axis, sign) maximizing dot(n_neighbor_face, n_edit).
    """
    U_rows = _axes_rows(neighbor_obb["axes"])
    n_edit = _normalize(_as_np(edit_face_normal_world))

    cands: List[Dict[str, Any]] = []
    for axis in (0, 1, 2):
        u = _normalize(U_rows[axis])
        for sign in (+1, -1):
            n_face = float(sign) * u
            score = float(np.dot(n_face, n_edit))
            cands.append({"axis": int(axis), "sign": int(sign), "score": float(score)})

    cands.sort(key=lambda d: d["score"], reverse=True)
    best = cands[0]
    return {
        "axis": int(best["axis"]),
        "sign": int(best["sign"]),
        "score": float(best["score"]),
        "candidates": cands,
    }


# ----------------------------
# Open3D visualization helpers
# ----------------------------

def _obb_to_o3d_obb(obb: Dict[str, Any]) -> o3d.geometry.OrientedBoundingBox:
    C = _as_np(obb["center"])
    U_rows = _axes_rows(obb["axes"])
    R_cols = U_rows.T  # open3d expects columns = axes
    ext_half = _as_np(obb["extents"])
    ext_full = 2.0 * ext_half
    return o3d.geometry.OrientedBoundingBox(C, R_cols, ext_full)


def _lineset_from_obb(obb: Dict[str, Any]) -> o3d.geometry.LineSet:
    return o3d.geometry.LineSet.create_from_oriented_bounding_box(_obb_to_o3d_obb(obb))


def _face_corners_world(obb: Dict[str, Any], axis: int, sign: int) -> np.ndarray:
    C = _as_np(obb["center"])
    U = _axes_rows(obb["axes"])
    E = _as_np(obb["extents"])

    axis = int(axis)
    sign = +1 if int(sign) >= 0 else -1

    uA = _normalize(U[axis])
    other = [0, 1, 2]
    other.remove(axis)
    i, j = other[0], other[1]
    ui = _normalize(U[i])
    uj = _normalize(U[j])

    fc = C + float(sign) * float(E[axis]) * uA
    di = float(E[i]) * ui
    dj = float(E[j]) * uj

    p0 = fc - di - dj
    p1 = fc + di - dj
    p2 = fc + di + dj
    p3 = fc - di + dj
    return np.stack([p0, p1, p2, p3], axis=0)


def _face_mesh(obb: Dict[str, Any], axis: int, sign: int) -> o3d.geometry.TriangleMesh:
    corners = _face_corners_world(obb, axis, sign)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners.tolist())
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.compute_vertex_normals()
    return mesh


def _face_outline(obb: Dict[str, Any], axis: int, sign: int) -> o3d.geometry.LineSet:
    corners = _face_corners_world(obb, axis, sign)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners.tolist())
    ls.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]])
    return ls


def vis_verify_inferred_edit_face_and_neighbor_face(
    target_before_obb: Dict[str, Any],
    target_after_obb: Dict[str, Any],
    inferred_axis: int,
    inferred_sign: int,
    neighbor_obb: Dict[str, Any],
    neighbor_axis: int,
    neighbor_sign: int,
    show_neighbor_opposite: bool = True,
) -> None:
    geoms: List[o3d.geometry.Geometry] = []

    # target OBBs
    ls_t0 = _lineset_from_obb(target_before_obb)
    ls_t0.paint_uniform_color([0.2, 0.4, 1.0])  # blue
    geoms.append(ls_t0)

    ls_t1 = _lineset_from_obb(target_after_obb)
    ls_t1.paint_uniform_color([1.0, 0.2, 0.2])  # red
    geoms.append(ls_t1)

    # inferred edited face (on BLUE)
    f0m = _face_mesh(target_before_obb, inferred_axis, inferred_sign)
    f0m.paint_uniform_color([0.2, 0.4, 1.0])
    geoms.append(f0m)
    f0o = _face_outline(target_before_obb, inferred_axis, inferred_sign)
    f0o.paint_uniform_color([0.2, 0.4, 1.0])
    geoms.append(f0o)

    # same face on RED (to see where it moved)
    f1m = _face_mesh(target_after_obb, inferred_axis, inferred_sign)
    f1m.paint_uniform_color([1.0, 0.2, 0.2])
    geoms.append(f1m)
    f1o = _face_outline(target_after_obb, inferred_axis, inferred_sign)
    f1o.paint_uniform_color([1.0, 0.2, 0.2])
    geoms.append(f1o)

    # neighbor OBB
    ls_b = _lineset_from_obb(neighbor_obb)
    ls_b.paint_uniform_color([0.2, 0.9, 0.2])  # green
    geoms.append(ls_b)

    # neighbor face to edit
    fbm = _face_mesh(neighbor_obb, neighbor_axis, neighbor_sign)
    fbm.paint_uniform_color([1.0, 1.0, 0.0])  # yellow
    geoms.append(fbm)
    fbo = _face_outline(neighbor_obb, neighbor_axis, neighbor_sign)
    fbo.paint_uniform_color([1.0, 1.0, 0.0])
    geoms.append(fbo)

    if show_neighbor_opposite:
        opp = _face_outline(neighbor_obb, neighbor_axis, -int(neighbor_sign))
        opp.paint_uniform_color([0.7, 0.7, 0.7])
        geoms.append(opp)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="AEP: inferred edited face (blue-only) + neighbor face to edit",
    )


# ----------------------------
# Public entry called by attachment.py (NO attachment.py changes)
# ----------------------------

def scale_neighbor_obb(
    other_obb: Dict[str, Any],
    edit_face_normal: np.ndarray,
    scale_ratio: float,
    min_extent: float,
    edited_face_str: str,
    neighbor_name: str,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Called by attachment.py.

    We IGNORE `edited_face_str` for determining the edited face, per your definition.
    We infer the edited face from target_before vs target_after geometry.

    For now:
      - NO scaling (return other_obb unchanged)
      - DO:
          * infer edited face from before/after (blue-only face)
          * choose neighbor face by aligning normals
          * open Open3D vis

    Returns:
      (other_obb unchanged), debug_info
    """
    # 1) reconstruct target before/after from stack
    ed = _find_edit_decomp_in_stack()
    if ed is None:
        # Cannot do your "blue-only face" definition without before/after.
        # Fallback: pick neighbor face based on provided normal, and skip vis.
        sel = choose_neighbor_face_to_change(other_obb, edit_face_normal)
        if verbose:
            print("[AEP][attachment_scaling] WARNING: could not find edit_decomp in stack -> cannot infer blue-only face.")
            print("[AEP][attachment_scaling] fallback: using provided edit_face_normal only (may be wrong for your definition).")
            print(f"[AEP][attachment_scaling] neighbor={neighbor_name}  chosen_neighbor_face={_face_to_str(sel['axis'], sel['sign'])}  score={sel['score']:.6f}")
        debug_info = {
            "status": "fallback_no_edit_decomp",
            "neighbor": neighbor_name,
            "chosen_neighbor_face_axis": int(sel["axis"]),
            "chosen_neighbor_face_sign": int(sel["sign"]),
            "chosen_neighbor_face_score": float(sel["score"]),
            "note": "could not infer edited face from before/after",
        }
        return other_obb, debug_info

    target_before_obb, target_after_obb = _reconstruct_target_obbs(ed)

    # 2) infer the edited face (blue-only face) from before/after
    inf = infer_edited_face_from_before_after(target_before_obb, target_after_obb)
    axis = int(inf["axis"])
    sign = int(inf["sign"])
    n_edit = _as_np(inf["normal_world"])

    # 3) choose neighbor face to edit using that inferred normal
    sel = choose_neighbor_face_to_change(other_obb, n_edit)

    if verbose:
        print("[AEP][attachment_scaling] scaling called! (NO-OP scaling for now)")
        print(f"[AEP][attachment_scaling] neighbor = {neighbor_name}")
        print(f"[AEP][attachment_scaling] inferred_target_edited_face(blue-only) = {_face_to_str(axis, sign)}  delta_plane={inf['delta_plane']:.6f}")
        print(f"[AEP][attachment_scaling] inferred_edit_normal = {n_edit.tolist()}")
        print(f"[AEP][attachment_scaling] chosen_neighbor_face_to_edit = {_face_to_str(sel['axis'], sel['sign'])}  score={sel['score']:.6f}")

        # quick sanity: show top-3 deltas and top-3 neighbor candidates
        top3d = inf["deltas"][:3]
        top3d_str = ", ".join([f"{_face_to_str(d['axis'], d['sign'])}:{d['delta']:+.6f}" for d in top3d])
        print(f"[AEP][attachment_scaling] top3_target_face_plane_deltas = {top3d_str}")

        top3n = sel["candidates"][:3]
        top3n_str = ", ".join([f"{_face_to_str(c['axis'], c['sign'])}:{c['score']:+.4f}" for c in top3n])
        print(f"[AEP][attachment_scaling] top3_neighbor_face_alignment = {top3n_str}")

        print("[AEP][attachment_scaling] opening Open3D verify window...")

    # 4) visualize (always when verbose=True)
    if verbose:
        vis_verify_inferred_edit_face_and_neighbor_face(
            target_before_obb=target_before_obb,
            target_after_obb=target_after_obb,
            inferred_axis=axis,
            inferred_sign=sign,
            neighbor_obb=other_obb,
            neighbor_axis=int(sel["axis"]),
            neighbor_sign=int(sel["sign"]),
            show_neighbor_opposite=True,
        )

    debug_info = {
        "status": "inferred_blue_only_face_and_selected_neighbor_face",
        "neighbor": neighbor_name,
        "inferred_target_face_axis": axis,
        "inferred_target_face_sign": sign,
        "inferred_target_face_str": _face_to_str(axis, sign),
        "inferred_target_face_delta_plane": float(inf["delta_plane"]),
        "inferred_target_normal_world": inf["normal_world"],
        "chosen_neighbor_face_axis": int(sel["axis"]),
        "chosen_neighbor_face_sign": int(sel["sign"]),
        "chosen_neighbor_face_str": _face_to_str(int(sel["axis"]), int(sel["sign"])),
        "chosen_neighbor_face_score": float(sel["score"]),
    }

    # no-op scaling for now
    return other_obb, debug_info
