# AEP/attachment_scaling.py
#
# Returns anchored neighbor scaling that follows the SAME PATTERN + SAME PORTION
# as the target BLUE(before)->RED(after) edit.
#
# - "edited face" = BLUE face plane that moved most between before->after
# - r = E_after[axis] / E_before[axis] along inferred axis
# - neighbor face to move chosen by aligning outward normals with inferred target edited face normal
# - neighbor update is ANCHORED: move that face, keep opposite face fixed, so center shifts
#
# IMPORTANT FIXES (per your vis.py):
# - OBBs in constraints are OBJECT SPACE. We do all math in OBJECT SPACE.
# - Visualization must transform object-space OBBs to WORLD SPACE using constraints["object_space"].
# - Open3D expects OBB axes as COLUMNS. Internally we keep OBB axes as COLUMNS everywhere.
#
# NOTE:
# - attachment.py stays unchanged except it calls scale_neighbor_obb(...)
# - This module uses stack inspection to reconstruct target before/after OBB (so no extra args needed)
# - It opens Open3D visualization when verbose=True (so you can verify)
#
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


def _face_to_str(axis: int, sign: int) -> str:
    s = "+" if int(sign) >= 0 else "-"
    return f"{s}u{int(axis)}"


# ----------------------------
# Stack inspection (no attachment.py changes)
# ----------------------------

def _find_edit_decomp_in_stack(max_depth: int = 16) -> Optional[Dict[str, Any]]:
    """
    We look for a dict-like edit_decomp/ed with keys:
      C0, U0, E0, C1, U1, E1

    We treat U0/U1 as (3,3) with COLUMNS as axes (Open3D-ready),
    matching vis.py assumptions.
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


def _find_object_space_in_stack(max_depth: int = 24) -> Optional[Dict[str, Any]]:
    """
    Try to find constraints["object_space"] (or a local named object_space) without
    changing attachment.py signature.

    We scan locals for:
      - "object_space" dict with keys "origin","axes"
      - "constraints" dict that contains "object_space"
    """
    frames = inspect.stack()
    try:
        for fr in frames[:max_depth]:
            loc = fr.frame.f_locals

            os_ = loc.get("object_space", None)
            if isinstance(os_, dict) and ("origin" in os_) and ("axes" in os_):
                return os_

            cons = loc.get("constraints", None)
            if isinstance(cons, dict):
                os2 = cons.get("object_space", None)
                if isinstance(os2, dict) and ("origin" in os2) and ("axes" in os2):
                    return os2
        return None
    finally:
        del frames


def _reconstruct_target_obbs(ed: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # keep axes as COLUMNS everywhere (Open3D-ready)
    t0 = {
        "center": _as_np(ed["C0"]).reshape(3,).tolist(),
        "axes": _as_np(ed["U0"]).reshape(3, 3).tolist(),      # columns are axes
        "extents": _as_np(ed["E0"]).reshape(3,).tolist(),     # half-lengths
    }
    t1 = {
        "center": _as_np(ed["C1"]).reshape(3,).tolist(),
        "axes": _as_np(ed["U1"]).reshape(3, 3).tolist(),      # columns are axes
        "extents": _as_np(ed["E1"]).reshape(3,).tolist(),     # half-lengths
    }
    return t0, t1


# ----------------------------
# Object-space -> world-space (for visualization ONLY)
# ----------------------------

def _get_object_space_T(object_space: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (origin_world, A_obj) where:
      origin_world: (3,)
      A_obj: (3,3) columns are object axes in world coordinates

    object_space json:
      {
        "origin": [..3..],
        "axes": [ [..3..], [..3..], [..3..] ]   # u0,u1,u2 vectors as ROWS
      }

    We convert list-of-vectors to a matrix with columns = u0,u1,u2.
    """
    origin = np.array(object_space["origin"], dtype=np.float64).reshape(3,)
    axes_list = np.array(object_space["axes"], dtype=np.float64)
    if axes_list.shape != (3, 3):
        raise ValueError("object_space['axes'] must be (3,3)")
    A_obj = axes_list.T
    return origin, A_obj


def _obb_object_to_world(obb_obj: Dict[str, Any], object_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an OBB defined in object space into world space.

    Assumptions:
      - obb_obj["center"] is in object coords
      - obb_obj["axes"] is a 3x3 basis with COLUMNS as local axes (Open3D-ready)
      - object_space provides object->world:
          p_world = origin + A_obj @ p_obj
        where A_obj columns are object axes in world.
    """
    origin, A_obj = _get_object_space_T(object_space)

    c_obj = np.array(obb_obj["center"], dtype=np.float64).reshape(3,)
    R_obj_cols = np.array(obb_obj["axes"], dtype=np.float64).reshape(3, 3)

    c_world = origin + (A_obj @ c_obj)
    R_world_cols = A_obj @ R_obj_cols

    return {
        "center": c_world.tolist(),
        "axes": R_world_cols.tolist(),   # columns are axes
        "extents": obb_obj["extents"],
    }


# ----------------------------
# Infer "edited face" from target before/after geometry (OBJECT SPACE)
# ----------------------------

def infer_edited_face_from_before_after(
    target_before_obb: Dict[str, Any],
    target_after_obb: Dict[str, Any],
) -> Dict[str, Any]:
    """
    "edited face" = BLUE face plane that moved most between before->after.

    All computations are in OBJECT SPACE.

    For a blue face (+/-ui), using BEFORE axes u_i (OBJECT SPACE):
      plane(i, s) = dot(u_i, C) + s * E_i
    delta(i, s) = plane_after - plane_before  (along same u_i)
    choose max abs(delta).
    """
    C0 = _as_np(target_before_obb["center"])
    R0 = _as_np(target_before_obb["axes"])   # columns
    E0 = _as_np(target_before_obb["extents"])

    C1 = _as_np(target_after_obb["center"])
    E1 = _as_np(target_after_obb["extents"])

    deltas: List[Dict[str, Any]] = []
    for i in (0, 1, 2):
        u = _normalize(R0[:, i])  # column i

        p0_plus = float(np.dot(u, C0) + float(E0[i]))
        p0_minus = float(np.dot(u, C0) - float(E0[i]))

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

    # normal in OBJECT SPACE (column axis, signed)
    n_obj = _normalize(sign * _normalize(R0[:, axis]))

    return {
        "axis": axis,
        "sign": sign,
        "normal_obj": n_obj.tolist(),
        "delta_plane": float(best["delta"]),
        "deltas": deltas,
    }


def target_scale_ratio_along_inferred_axis(
    target_before_obb: Dict[str, Any],
    target_after_obb: Dict[str, Any],
    axis: int,
) -> float:
    E0 = _as_np(target_before_obb["extents"])
    E1 = _as_np(target_after_obb["extents"])
    denom = float(max(E0[int(axis)], 1e-12))
    return float(E1[int(axis)] / denom)


# ----------------------------
# Neighbor face selection (normal alignment) (OBJECT SPACE)
# ----------------------------

def choose_neighbor_face_to_change(
    neighbor_obb: Dict[str, Any],
    edit_face_normal_obj: np.ndarray,
) -> Dict[str, Any]:
    """
    Choose neighbor face (+/-u_i) whose outward normal best aligns with
    the target edited face normal, all in OBJECT SPACE.

    Neighbor axes are columns in neighbor_obb["axes"].
    """
    R = _as_np(neighbor_obb["axes"])
    n_edit = _normalize(_as_np(edit_face_normal_obj))

    cands: List[Dict[str, Any]] = []
    for axis in (0, 1, 2):
        u = _normalize(R[:, axis])
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
# Apply anchored scaling on neighbor (OBJECT SPACE)
# ----------------------------

def apply_anchored_scale_on_neighbor(
    neighbor_before_obb: Dict[str, Any],
    neighbor_axis: int,
    neighbor_sign_move: int,
    r: float,
    min_extent: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Scale neighbor along neighbor_axis by ratio r, anchored so opposite face stays fixed.

    All in OBJECT SPACE.

    If s is sign of the MOVING face (+1 for +ui, -1 for -ui):
      extent: e -> e' = e*r
      center shift along u:  C' = C + s*(e' - e)*u
    """
    C = _as_np(neighbor_before_obb["center"])
    R = _as_np(neighbor_before_obb["axes"])  # columns
    E = _as_np(neighbor_before_obb["extents"]).copy()

    ax = int(neighbor_axis)
    s = +1 if int(neighbor_sign_move) >= 0 else -1
    u = _normalize(R[:, ax])

    e0 = float(E[ax])
    e1 = float(max(float(min_extent), e0 * float(r)))
    E[ax] = e1

    dproj = float(s) * (e1 - e0)
    C2 = C + dproj * u

    out = {"center": C2.tolist(), "axes": R.tolist(), "extents": E.tolist()}
    dbg = {
        "neighbor_axis": ax,
        "neighbor_sign_move": int(s),
        "r": float(r),
        "extent_before": float(e0),
        "extent_after": float(e1),
        "center_shift_obj": (C2 - C).tolist(),
        "anchored_fixed_face": _face_to_str(ax, -s),
        "moving_face": _face_to_str(ax, s),
    }
    return out, dbg


# ----------------------------
# Open3D visualization (WORLD SPACE) with object_space
# ----------------------------

def _obb_to_lineset(obb_world: Dict[str, Any]) -> o3d.geometry.LineSet:
    center = np.array(obb_world["center"], dtype=np.float64).reshape(3,)
    R = np.array(obb_world["axes"], dtype=np.float64).reshape(3, 3)  # columns
    ext_half = np.array(obb_world["extents"], dtype=np.float64).reshape(3,)
    ext_full = 2.0 * ext_half
    o3d_obb = o3d.geometry.OrientedBoundingBox(center, R, ext_full)
    return o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_obb)


def _paint_lineset(ls: o3d.geometry.LineSet, rgb) -> o3d.geometry.LineSet:
    colors = np.tile(np.array(rgb, dtype=np.float64)[None, :], (len(ls.lines), 1))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _face_corners_world_from_obj(
    obb_obj: Dict[str, Any],
    object_space: Dict[str, Any],
    axis: int,
    sign: int,
) -> np.ndarray:
    """
    Compute face corners in WORLD by:
      - compute corners in OBJECT space using OBB columns axes
      - map points to world using object_space
    """
    origin, A_obj = _get_object_space_T(object_space)

    C = _as_np(obb_obj["center"])
    R = _as_np(obb_obj["axes"])  # columns
    E = _as_np(obb_obj["extents"])

    axis = int(axis)
    sign = +1 if int(sign) >= 0 else -1

    uA = _normalize(R[:, axis])

    other = [0, 1, 2]
    other.remove(axis)
    i, j = other[0], other[1]
    ui = _normalize(R[:, i])
    uj = _normalize(R[:, j])

    fc = C + float(sign) * float(E[axis]) * uA
    di = float(E[i]) * ui
    dj = float(E[j]) * uj

    p0 = fc - di - dj
    p1 = fc + di - dj
    p2 = fc + di + dj
    p3 = fc - di + dj

    P_obj = np.stack([p0, p1, p2, p3], axis=0)  # (4,3)
    P_world = origin[None, :] + (P_obj @ A_obj.T)  # since p_world = origin + A_obj @ p_obj
    return P_world


def _face_mesh_world_from_obj(
    obb_obj: Dict[str, Any],
    object_space: Dict[str, Any],
    axis: int,
    sign: int,
) -> o3d.geometry.TriangleMesh:
    corners = _face_corners_world_from_obj(obb_obj, object_space, axis, sign)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners.tolist())
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.compute_vertex_normals()
    return mesh


def _face_outline_world_from_obj(
    obb_obj: Dict[str, Any],
    object_space: Dict[str, Any],
    axis: int,
    sign: int,
) -> o3d.geometry.LineSet:
    corners = _face_corners_world_from_obj(obb_obj, object_space, axis, sign)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners.tolist())
    ls.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]])
    return ls


def _vis_verify_neighbor_change(
    target_before_obj: Dict[str, Any],
    target_after_obj: Dict[str, Any],
    t_axis: int,
    t_sign: int,
    neighbor_before_obj: Dict[str, Any],
    neighbor_after_obj: Dict[str, Any],
    nb_axis: int,
    nb_sign: int,
    object_space: Optional[Dict[str, Any]],
) -> None:
    """
    Visual legend (WORLD space):
      - Target OBB:  BLUE(before), RED(after)
      - Neighbor OBB: GREEN(before), MAGENTA(after)
      - Target edited face: solid BLUE(before), solid RED(after)
      - Neighbor moved face: solid YELLOW(before), solid MAGENTA(after)
      - Neighbor fixed opposite face: gray outline (from BEFORE)
    """
    geoms: List[o3d.geometry.Geometry] = []

    # Convert OBBs to world for drawing lines
    if object_space is not None:
        t0w = _obb_object_to_world(target_before_obj, object_space)
        t1w = _obb_object_to_world(target_after_obj, object_space)
        nb0w = _obb_object_to_world(neighbor_before_obj, object_space)
        nb1w = _obb_object_to_world(neighbor_after_obj, object_space)
    else:
        # fallback: treat as world already
        t0w, t1w, nb0w, nb1w = target_before_obj, target_after_obj, neighbor_before_obj, neighbor_after_obj

    # Target lines (blue then red)
    geoms.append(_paint_lineset(_obb_to_lineset(t0w), (0.2, 0.4, 1.0)))
    geoms.append(_paint_lineset(_obb_to_lineset(t1w), (1.0, 0.2, 0.2)))

    # Neighbor lines (green then magenta)
    geoms.append(_paint_lineset(_obb_to_lineset(nb0w), (0.2, 0.9, 0.2)))
    geoms.append(_paint_lineset(_obb_to_lineset(nb1w), (1.0, 0.0, 1.0)))

    # Faces: only if we have object_space (otherwise face extraction would be ambiguous if world-space basis differs)
    if object_space is not None:
        # Target edited face (blue then red)
        f_t0 = _face_mesh_world_from_obj(target_before_obj, object_space, t_axis, t_sign)
        f_t0.paint_uniform_color([0.2, 0.4, 1.0])
        geoms.append(f_t0)

        f_t1 = _face_mesh_world_from_obj(target_after_obj, object_space, t_axis, t_sign)
        f_t1.paint_uniform_color([1.0, 0.2, 0.2])
        geoms.append(f_t1)

        # Neighbor moved face (yellow before, magenta after)
        f_nb0 = _face_mesh_world_from_obj(neighbor_before_obj, object_space, nb_axis, nb_sign)
        f_nb0.paint_uniform_color([1.0, 1.0, 0.0])
        geoms.append(f_nb0)

        f_nb1 = _face_mesh_world_from_obj(neighbor_after_obj, object_space, nb_axis, nb_sign)
        f_nb1.paint_uniform_color([1.0, 0.0, 1.0])
        geoms.append(f_nb1)

        # Fixed opposite face outline (from neighbor BEFORE)
        opp = _face_outline_world_from_obj(neighbor_before_obj, object_space, nb_axis, -int(nb_sign))
        opp.paint_uniform_color([0.65, 0.65, 0.65])
        geoms.append(opp)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="AEP verify (object->world): target blue->red, neighbor green->magenta, faces highlighted",
    )


# ----------------------------
# Public entry called by attachment.py
# ----------------------------

def scale_neighbor_obb(
    other_obb: Dict[str, Any],
    edit_face_normal: np.ndarray,  # kept for compatibility; we infer from target pattern
    scale_ratio: float,            # kept for compatibility; we recompute r from target blue->red
    min_extent: float,
    edited_face_str: str,          # kept for compatibility; we infer from target pattern
    neighbor_name: str,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      (neighbor_after_obb_obj, debug_info)

    All geometry updates are done in OBJECT SPACE (consistent with constraints storage).
    Visualization (if verbose) converts to WORLD SPACE using constraints["object_space"].
    """
    ed = _find_edit_decomp_in_stack()
    if ed is None:
        if verbose:
            print("[AEP][attachment_scaling] ERROR: cannot find edit_decomp in stack -> cannot compute target pattern.")
        return other_obb, {"status": "error_no_edit_decomp", "neighbor": neighbor_name}

    object_space = _find_object_space_in_stack()

    target_before_obb, target_after_obb = _reconstruct_target_obbs(ed)

    # 1) infer edited face from target geometry (OBJECT SPACE)
    inf = infer_edited_face_from_before_after(target_before_obb, target_after_obb)
    t_axis = int(inf["axis"])
    t_sign = int(inf["sign"])
    n_edit_obj = _as_np(inf["normal_obj"])

    # 2) compute portion ratio r from target extents along inferred axis (OBJECT SPACE)
    r = target_scale_ratio_along_inferred_axis(target_before_obb, target_after_obb, t_axis)

    # 3) choose neighbor face to move by normal alignment (OBJECT SPACE)
    sel = choose_neighbor_face_to_change(other_obb, n_edit_obj)
    nb_axis = int(sel["axis"])
    nb_sign = int(sel["sign"])

    # 4) anchored neighbor scaling (OBJECT SPACE)
    neighbor_before_obb = other_obb
    neighbor_after_obb, dbg_scale = apply_anchored_scale_on_neighbor(
        neighbor_before_obb,
        neighbor_axis=nb_axis,
        neighbor_sign_move=nb_sign,
        r=r,
        min_extent=float(min_extent),
    )

    if verbose:
        print("[AEP][attachment_scaling] scaling called!")
        print(f"[AEP][attachment_scaling] neighbor = {neighbor_name}")
        print(f"[AEP][attachment_scaling] inferred_target_edited_face(blue-only) = {_face_to_str(t_axis, t_sign)}  delta_plane={inf['delta_plane']:+.6f}")
        print(f"[AEP][attachment_scaling] target_ratio r = {r:.6f}  (E_after/E_before along axis={t_axis})")
        print(f"[AEP][attachment_scaling] chosen_neighbor_face_to_MOVE = {_face_to_str(nb_axis, nb_sign)}  score={sel['score']:+.6f}")
        print(f"[AEP][attachment_scaling] anchored: move={dbg_scale['moving_face']} keep_fixed={dbg_scale['anchored_fixed_face']}")
        print(f"[AEP][attachment_scaling]   extent axis={nb_axis}: {dbg_scale['extent_before']:.6f} -> {dbg_scale['extent_after']:.6f}")
        print(f"[AEP][attachment_scaling]   center shift (OBJECT): {dbg_scale['center_shift_obj']}")
        if object_space is None:
            print("[AEP][attachment_scaling] WARN: object_space not found in stack -> visualization will be object-space-as-world.")
        print("[AEP][attachment_scaling] opening Open3D verify window...")

        # _vis_verify_neighbor_change(
        #     target_before_obj=target_before_obb,
        #     target_after_obj=target_after_obb,
        #     t_axis=t_axis,
        #     t_sign=t_sign,
        #     neighbor_before_obj=neighbor_before_obb,
        #     neighbor_after_obj=neighbor_after_obb,
        #     nb_axis=nb_axis,
        #     nb_sign=nb_sign,
        #     object_space=object_space,
        # )

    debug_info = {
        "status": "neighbor_scaled_anchored",
        "neighbor": neighbor_name,
        "target_inferred_face": _face_to_str(t_axis, t_sign),
        "target_inferred_face_delta_plane": float(inf["delta_plane"]),
        "target_ratio_r": float(r),
        "neighbor_face_moved": _face_to_str(nb_axis, nb_sign),
        "neighbor_align_score": float(sel["score"]),
        "neighbor_scale_debug": dbg_scale,
        "has_object_space_for_vis": bool(object_space is not None),
    }

    # IMPORTANT: return OBJECT-SPACE neighbor OBB (consistent with constraints storage)
    return neighbor_after_obb, debug_info
