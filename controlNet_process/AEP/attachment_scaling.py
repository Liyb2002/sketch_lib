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
# NOTE:
# - attachment.py stays unchanged except it calls scale_neighbor_obb(...)
# - This module uses stack inspection to reconstruct target before/after OBB (so no extra args needed)
# - It also opens Open3D visualization when verbose=True (so you can verify)
#
# No __main__ section.

from typing import Dict, Any, Tuple, List, Optional
import inspect
import numpy as np
import open3d as o3d


# ----------------------------
# Basic helpers
# ----------------------------

verbose=False

def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _axes_rows(axes_list) -> np.ndarray:
    # axes stored as ROWS (u0,u1,u2)
    return _as_np(axes_list)


def _face_to_str(axis: int, sign: int) -> str:
    s = "+" if int(sign) >= 0 else "-"
    return f"{s}u{int(axis)}"


# ----------------------------
# Stack inspection (no attachment.py changes)
# ----------------------------

def _find_edit_decomp_in_stack(max_depth: int = 12) -> Optional[Dict[str, Any]]:
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
# Infer "edited face" from target before/after geometry
# ----------------------------

def infer_edited_face_from_before_after(
    target_before_obb: Dict[str, Any],
    target_after_obb: Dict[str, Any],
) -> Dict[str, Any]:
    """
    "edited face" = BLUE face plane that moved most between before->after.

    For a blue face (+/-ui), using BEFORE axes u:
      plane(i, s) = dot(u_i, C) + s * E_i
    delta(i, s) = plane_after - plane_before  (projected on same u_i)
    choose max abs(delta).
    """
    C0 = _as_np(target_before_obb["center"])
    U0 = _axes_rows(target_before_obb["axes"])
    E0 = _as_np(target_before_obb["extents"])

    C1 = _as_np(target_after_obb["center"])
    E1 = _as_np(target_after_obb["extents"])

    deltas: List[Dict[str, Any]] = []
    for i in (0, 1, 2):
        u = _normalize(U0[i])

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

    n_world = _normalize(sign * _normalize(U0[axis]))

    return {
        "axis": axis,
        "sign": sign,
        "normal_world": n_world.tolist(),
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
# Neighbor face selection (normal alignment)
# ----------------------------

def choose_neighbor_face_to_change(
    neighbor_obb: Dict[str, Any],
    edit_face_normal_world: np.ndarray,
) -> Dict[str, Any]:
    U = _axes_rows(neighbor_obb["axes"])
    n_edit = _normalize(_as_np(edit_face_normal_world))

    cands: List[Dict[str, Any]] = []
    for axis in (0, 1, 2):
        u = _normalize(U[axis])
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
# Apply anchored scaling on neighbor
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

    If s is sign of the MOVING face (+1 for +ui, -1 for -ui):
      extent: e -> e' = e*r
      center shift along u:  C' = C + s*(e' - e)*u
    """
    C = _as_np(neighbor_before_obb["center"])
    U = _axes_rows(neighbor_before_obb["axes"])
    E = _as_np(neighbor_before_obb["extents"]).copy()

    ax = int(neighbor_axis)
    s = +1 if int(neighbor_sign_move) >= 0 else -1
    u = _normalize(U[ax])

    e0 = float(E[ax])
    e1 = float(max(float(min_extent), e0 * float(r)))
    E[ax] = e1

    dproj = float(s) * (e1 - e0)
    C2 = C + dproj * u

    out = {"center": C2.tolist(), "axes": U.tolist(), "extents": E.tolist()}
    dbg = {
        "neighbor_axis": ax,
        "neighbor_sign_move": int(s),
        "r": float(r),
        "extent_before": float(e0),
        "extent_after": float(e1),
        "center_shift_world": (C2 - C).tolist(),
        "anchored_fixed_face": _face_to_str(ax, -s),
        "moving_face": _face_to_str(ax, s),
    }
    return out, dbg


# ----------------------------
# Open3D visualization (verify)
# ----------------------------

def _obb_to_o3d_obb(obb: Dict[str, Any]) -> o3d.geometry.OrientedBoundingBox:
    C = _as_np(obb["center"])
    U_rows = _axes_rows(obb["axes"])
    R_cols = U_rows.T
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


def _vis_verify_neighbor_change(
    target_before_obb: Dict[str, Any],
    target_after_obb: Dict[str, Any],
    inferred_axis: int,
    inferred_sign: int,
    neighbor_before_obb: Dict[str, Any],
    neighbor_after_obb: Dict[str, Any],
    neighbor_axis: int,
    neighbor_sign: int,
) -> None:
    geoms: List[o3d.geometry.Geometry] = []

    ls_t0 = _lineset_from_obb(target_before_obb)
    ls_t0.paint_uniform_color([0.2, 0.4, 1.0])
    geoms.append(ls_t0)

    ls_t1 = _lineset_from_obb(target_after_obb)
    ls_t1.paint_uniform_color([1.0, 0.2, 0.2])
    geoms.append(ls_t1)

    f0m = _face_mesh(target_before_obb, inferred_axis, inferred_sign)
    f0m.paint_uniform_color([0.2, 0.4, 1.0])
    geoms.append(f0m)

    f1m = _face_mesh(target_after_obb, inferred_axis, inferred_sign)
    f1m.paint_uniform_color([1.0, 0.2, 0.2])
    geoms.append(f1m)

    ls_nb = _lineset_from_obb(neighbor_before_obb)
    ls_nb.paint_uniform_color([0.2, 0.9, 0.2])
    geoms.append(ls_nb)

    ls_na = _lineset_from_obb(neighbor_after_obb)
    ls_na.paint_uniform_color([1.0, 0.0, 1.0])  # magenta
    geoms.append(ls_na)

    nb_face = _face_mesh(neighbor_before_obb, neighbor_axis, neighbor_sign)
    nb_face.paint_uniform_color([1.0, 1.0, 0.0])
    geoms.append(nb_face)

    na_face = _face_mesh(neighbor_after_obb, neighbor_axis, neighbor_sign)
    na_face.paint_uniform_color([1.0, 0.0, 1.0])
    geoms.append(na_face)

    opp = _face_outline(neighbor_before_obb, neighbor_axis, -int(neighbor_sign))
    opp.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(opp)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="AEP: target edit pattern + neighbor anchored scaling",
    )


# ----------------------------
# Public entry called by attachment.py
# ----------------------------

def scale_neighbor_obb(
    other_obb: Dict[str, Any],
    edit_face_normal: np.ndarray,  # kept for compatibility, not used for inference
    scale_ratio: float,            # kept for compatibility, we recompute r from target blue->red
    min_extent: float,
    edited_face_str: str,          # kept for compatibility, not used for inference
    neighbor_name: str,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ed = _find_edit_decomp_in_stack()
    if ed is None:
        if verbose:
            print("[AEP][attachment_scaling] ERROR: cannot find edit_decomp in stack -> cannot compute target pattern.")
        return other_obb, {"status": "error_no_edit_decomp", "neighbor": neighbor_name}

    target_before_obb, target_after_obb = _reconstruct_target_obbs(ed)

    # 1) infer edited face from target geometry (blue-only face)
    inf = infer_edited_face_from_before_after(target_before_obb, target_after_obb)
    t_axis = int(inf["axis"])
    t_sign = int(inf["sign"])
    n_edit = _as_np(inf["normal_world"])

    # 2) compute portion ratio r from target extents along inferred axis
    r = target_scale_ratio_along_inferred_axis(target_before_obb, target_after_obb, t_axis)

    # 3) choose neighbor face to move by normal alignment
    sel = choose_neighbor_face_to_change(other_obb, n_edit)
    nb_axis = int(sel["axis"])
    nb_sign = int(sel["sign"])

    # 4) anchored neighbor scaling
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
        print(f"[AEP][attachment_scaling] target_ratio r = {r:.6f}  (E_after/E_before along u{t_axis})")
        print(f"[AEP][attachment_scaling] chosen_neighbor_face_to_MOVE = {_face_to_str(nb_axis, nb_sign)}  score={sel['score']:+.6f}")
        print(f"[AEP][attachment_scaling] anchored: move={dbg_scale['moving_face']} keep_fixed={dbg_scale['anchored_fixed_face']}")
        print(f"[AEP][attachment_scaling]   extent u{nb_axis}: {dbg_scale['extent_before']:.6f} -> {dbg_scale['extent_after']:.6f}")
        print(f"[AEP][attachment_scaling]   center shift: {dbg_scale['center_shift_world']}")
        print("[AEP][attachment_scaling] opening Open3D verify window...")

        # _vis_verify_neighbor_change(
        #     target_before_obb=target_before_obb,
        #     target_after_obb=target_after_obb,
        #     inferred_axis=t_axis,
        #     inferred_sign=t_sign,
        #     neighbor_before_obb=neighbor_before_obb,
        #     neighbor_after_obb=neighbor_after_obb,
        #     neighbor_axis=nb_axis,
        #     neighbor_sign=nb_sign,
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
    }

    return neighbor_after_obb, debug_info
