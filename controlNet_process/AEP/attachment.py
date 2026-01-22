#!/usr/bin/env python3
# AEP/attachment.py
#
# Computes attachment-driven AEP propagation (per your finalized rules),
# and prints (when verbose=True), for EACH edge touching the target:
#   1) attachment type (volume/face/point)
#   2) where the attachment is relative to the edited face
#   3) operation: translation / scaling / none  (never both in current rules)
#
# IMPORTANT:
# - Keep same public API:
#     apply_attachments(constraints, edit, verbose=True) -> attach_res dict
# - No __main__ section
#
# Assumptions:
# - extents are half-lengths
# - axes are (approx) orthonormal
# - edit["change"] contains before_obb and after_obb for the target
# - constraints["nodes"][name]["obb"] exists for each component
#
# Behavior summary:
#   FACE:
#     A1 (same face): translate B by edited face displacement ΔF
#     A2 (opposite):  no change
#     A3 (perp):      scaling policy (delegated to attachment_scaling.py)
#
#   VOLUME:
#     Infer which TARGET face (+/-u0/u1/u2) is closest to the volume contact box
#     (in A-before local coords), then compare vs edited face:
#       V1 (same):       translate by ΔF
#       V2 (opposite):   no change
#       V3 (perp):       scaling policy (delegated to attachment_scaling.py)
#     (Fallback: if no local box, use anchor_world as degenerate box; if neither, scale policy.)
#
#   POINT:
#     Preserve relation "box <-> point" by:
#       P1: move anchor to preserve A-point relation (closest face + offset + tangential coords)
#       P2: translate B so that B-point relation is unchanged (rigid: ΔB = P1 - P0)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Scaling is delegated to a separate module to keep this file simpler.
# For now, the scaling module is a stub that prints "scaling called!" and returns unchanged OBB.
from .attachment_scaling import scale_neighbor_obb


# ----------------------------
# Basic helpers
# ----------------------------

def _as_np(x):
    return np.asarray(x, dtype=np.float64)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def _axes_matrix(axes_list: List[List[float]]) -> np.ndarray:
    """
    Your JSON stores axes as 3 vectors. We'll treat them as rows (u0,u1,u2),
    and use dot with each axis for local coords.
    """
    U = _as_np(axes_list)  # (3,3)
    return U


def _world_to_local(P: np.ndarray, C: np.ndarray, U_rows: np.ndarray) -> np.ndarray:
    # local coords = [dot(P-C, u0), dot(P-C, u1), dot(P-C, u2)]
    d = P - C
    return U_rows @ d


def _local_to_world(q: np.ndarray, C: np.ndarray, U_rows: np.ndarray) -> np.ndarray:
    # P = C + sum_i q_i * u_i
    return C + U_rows.T @ q


def _parse_face_str(face: Any) -> Optional[Tuple[int, int]]:
    """
    Parse "+u0" / "-u2" (or "+0"/"-1" variants if they appear)
    Returns (axis_index, sign)
    """
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
        # no sign, assume +
        sign = +1
        s2 = s

    # expected "u0","u1","u2"
    if s2.startswith("u") and len(s2) >= 2 and s2[1].isdigit():
        axis = int(s2[1])
        if axis in (0, 1, 2):
            return axis, sign

    # fallback: "0","1","2"
    if s2.isdigit():
        axis = int(s2)
        if axis in (0, 1, 2):
            return axis, sign

    return None


def _face_to_str(axis: int, sign: int) -> str:
    s = "+" if int(sign) >= 0 else "-"
    return f"{s}u{int(axis)}"


def _infer_attachment_kind(e: Dict[str, Any]) -> str:
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

    if any(x in e for x in [
        "overlap_volume", "vol_overlap",
        "overlap_volume_est", "overlap_frac_small",
        "overlap_box_local_min", "overlap_box_local_max",
        "vol_a", "vol_b",
    ]):
        return "volume"

    if ("a_face" in e) and ("b_face" in e):
        return "face"

    return "point"


def _other_name_in_edge(e: Dict[str, Any], name: str) -> Optional[str]:
    a = e.get("a", None)
    b = e.get("b", None)
    if a == name and isinstance(b, str):
        return b
    if b == name and isinstance(a, str):
        return a
    return None


def _get_node_obb(constraints: Dict[str, Any], name: str) -> Dict[str, Any]:
    nodes = constraints.get("nodes", {}) or {}
    if name not in nodes:
        raise KeyError(f"constraints['nodes'] missing '{name}'")
    obb = nodes[name].get("obb", None)
    if not isinstance(obb, dict):
        raise KeyError(f"constraints['nodes'][{name}]['obb'] missing/invalid")
    for k in ["center", "axes", "extents"]:
        if k not in obb:
            raise KeyError(f"obb missing '{k}' for node '{name}'")
    return obb


def _min_extent_from_edit(change: Dict[str, Any]) -> float:
    me = change.get("min_extent", None)
    if isinstance(me, (int, float)):
        return float(me)
    return 1e-4


# ----------------------------
# Edit decomposition helpers
# ----------------------------

def _face_center(C: np.ndarray, U_rows: np.ndarray, E: np.ndarray, axis: int, sign: int) -> np.ndarray:
    # F = C + sign * E[axis] * u_axis
    return C + float(sign) * float(E[axis]) * U_rows[axis]


def _compute_edit_decomp(change: Dict[str, Any]) -> Dict[str, Any]:
    before = change.get("before_obb", None)
    after = change.get("after_obb", None)
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("edit['change'] missing before_obb/after_obb")

    C0 = _as_np(before["center"])
    U0 = _axes_matrix(before["axes"])
    E0 = _as_np(before["extents"])

    C1 = _as_np(after["center"])
    U1 = _axes_matrix(after["axes"])
    E1 = _as_np(after["extents"])

    axis = int(change.get("axis", -1))
    face_str = change.get("face", None)
    parsed = _parse_face_str(face_str)
    if parsed is None:
        s_edit = int(change.get("sign", +1))
        s_edit = +1 if s_edit >= 0 else -1
    else:
        k_face, s_edit = parsed
        # if axis mismatch, trust "axis"
        _ = k_face

    if axis not in (0, 1, 2):
        raise ValueError(f"Invalid edit axis: {axis}")

    # scale ratio along edited axis
    r = float(E1[axis] / max(E0[axis], 1e-12))

    # face displacement for edited face
    F0 = _face_center(C0, U0, E0, axis, s_edit)
    F1 = _face_center(C1, U1, E1, axis, s_edit)
    dF = F1 - F0

    dC = C1 - C0
    uk = _normalize(U0[axis])

    # outward normal of edited face (signed)
    n_edit = _normalize(float(s_edit) * U0[axis])

    return {
        "C0": C0, "U0": U0, "E0": E0,
        "C1": C1, "U1": U1, "E1": E1,
        "axis": axis,
        "s_edit": int(s_edit),
        "r": float(r),
        "dF": dF,
        "dC": dC,
        "uk": uk,
        "n_edit": n_edit,
    }


def _apply_translation(obb: Dict[str, Any], d_world: np.ndarray) -> Dict[str, Any]:
    C = _as_np(obb["center"])
    U = _axes_matrix(obb["axes"])
    E = _as_np(obb["extents"])
    C2 = C + d_world
    return {"center": C2.tolist(), "axes": U.tolist(), "extents": E.tolist()}


# ----------------------------
# Pretty-print helpers
# ----------------------------

def _print_edge_header(kind: str, target: str, other: str, idx: int):
    print(f"\n[AEP][attach] edge#{idx}  A(target)={target}  B(neighbor)={other}")
    print(f"[AEP][attach] 1) attachment_type = {kind}")


def _print_where_and_op(where: str, op: str):
    # op must be: "translation" | "scaling" | "none"
    print(f"[AEP][attach] 2) where_vs_edited_face = {where}")
    print(f"[AEP][attach] 3) operation = {op}")


# ----------------------------
# Point attachment: preserve box<->point relation (UNCHANGED)
# ----------------------------

def _closest_face_relation_local(q: np.ndarray, E: np.ndarray) -> Tuple[int, int, float]:
    """
    Returns:
      axis i*, sign s* (+1 for +ui face, -1 for -ui face),
      and offset delta = q[i*] - s*E[i*]  (signed wrt that plane)
    """
    best = None
    best_i = 0
    best_s = +1
    best_abs = 1e30
    for i in range(3):
        # distance to +face plane (qi=+Ei)
        d_plus = float(q[i] - E[i])
        a_plus = abs(d_plus)
        if a_plus < best_abs:
            best_abs = a_plus
            best_i = i
            best_s = +1
            best = d_plus
        # distance to -face plane (qi=-Ei)
        d_minus = float(q[i] + E[i])
        a_minus = abs(d_minus)
        if a_minus < best_abs:
            best_abs = a_minus
            best_i = i
            best_s = -1
            best = d_minus
    delta = float(best if best is not None else 0.0)
    return best_i, best_s, delta


def _move_anchor_preserve_A_relation(P0: np.ndarray, edit_decomp: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preserve:
      - closest face id (axis + sign)
      - offset delta to that face plane
      - tangential local coords
    """
    C0, U0, E0 = edit_decomp["C0"], edit_decomp["U0"], edit_decomp["E0"]
    C1, U1, E1 = edit_decomp["C1"], edit_decomp["U1"], edit_decomp["E1"]

    q0 = _world_to_local(P0, C0, U0)
    i_star, s_star, delta = _closest_face_relation_local(q0, E0)

    q1 = q0.copy()
    q1[i_star] = float(s_star * E1[i_star] + delta)

    P1 = _local_to_world(q1, C1, U1)

    info = {
        "A_face_axis": int(i_star),
        "A_face_sign": int(s_star),
        "A_delta": float(delta),
        "P0_local": q0.tolist(),
        "P1_local": q1.tolist(),
    }
    return P1, info


# ----------------------------
# Face attachment: A1/A2/A3 (scaling delegated)
# ----------------------------

def _solve_face_edge(
    target: str,
    other: str,
    e: Dict[str, Any],
    edit_decomp: Dict[str, Any],
    other_obb: Dict[str, Any],
    min_extent: float,
    verbose: bool
) -> Optional[Dict[str, Any]]:
    """
    Returns a change record for 'other' or None.
    """
    # Determine which face string corresponds to the target in this edge
    if e.get("a") == target:
        t_face = e.get("a_face", None)
        o_face = e.get("b_face", None)
    else:
        t_face = e.get("b_face", None)
        o_face = e.get("a_face", None)

    t_parsed = _parse_face_str(t_face)
    k = int(edit_decomp["axis"])
    s_edit = int(edit_decomp["s_edit"])
    edited_face_str = _face_to_str(k, s_edit)

    if t_parsed is None:
        # can't classify; fall back to "perpendicular -> scaling policy"
        solving = "A3(face_fallback_no_face)"
        where = f"unknown_target_face -> treat_as_perpendicular_to_edited_face({edited_face_str})"

        after_obb, scale_debug = scale_neighbor_obb(
            other_obb=other_obb,
            edit_face_normal=edit_decomp["n_edit"],
            scale_ratio=edit_decomp["r"],
            min_extent=min_extent,
            edited_face_str=edited_face_str,
            neighbor_name=other,
            verbose=verbose,
        )

        if verbose:
            _print_where_and_op(where=where, op="scaling")
            print(f"[AEP][attach][FACE] solving={solving} | delegated_scaling=stub (no actual change)")

        return {
            "kind": "face",
            "solving": solving,
            "edge": {"a": e.get("a"), "b": e.get("b"), "a_face": e.get("a_face"), "b_face": e.get("b_face")},
            "before_obb": other_obb,
            "after_obb": after_obb,
            "op": {
                "type": "scale_stub",
                "ratio": float(edit_decomp["r"]),
                "debug": scale_debug,
            },
        }

    t_axis, t_sign = t_parsed

    if t_axis == k and t_sign == s_edit:
        # A1
        solving = "A1"
        where = f"attached_on_same_face_as_edit: target_face={_face_to_str(t_axis, t_sign)} == edited_face={edited_face_str}"
        after_obb = _apply_translation(other_obb, edit_decomp["dF"])
        if verbose:
            _print_where_and_op(where=where, op="translation")
            print(f"[AEP][attach][FACE] solving={solving} | delta(dF)={edit_decomp['dF'].tolist()}")
        return {
            "kind": "face",
            "solving": solving,
            "edge": {"a": e.get("a"), "b": e.get("b"), "a_face": e.get("a_face"), "b_face": e.get("b_face")},
            "before_obb": other_obb,
            "after_obb": after_obb,
            "op": {"type": "translate", "delta_world": edit_decomp["dF"].tolist()},
        }

    if t_axis == k and t_sign == -s_edit:
        # A2
        solving = "A2"
        where = f"attached_on_opposite_face: target_face={_face_to_str(t_axis, t_sign)} opposite_of edited_face={edited_face_str}"
        if verbose:
            _print_where_and_op(where=where, op="none")
            print(f"[AEP][attach][FACE] solving={solving} | no change")
        return None

    # perpendicular -> A3 (delegate scaling)
    solving = "A3"
    where = f"attached_on_perpendicular_face: target_face={_face_to_str(t_axis, t_sign)} perp_to edited_face={edited_face_str}"

    after_obb, scale_debug = scale_neighbor_obb(
        other_obb=other_obb,
        edit_face_normal=edit_decomp["n_edit"],
        scale_ratio=edit_decomp["r"],
        min_extent=min_extent,
        edited_face_str=edited_face_str,
        neighbor_name=other,
        verbose=verbose,
    )

    if verbose:
        _print_where_and_op(where=where, op="scaling")
        print(f"[AEP][attach][FACE] solving={solving} | delegated_scaling=stub (no actual change)")

    return {
        "kind": "face",
        "solving": solving,
        "edge": {"a": e.get("a"), "b": e.get("b"), "a_face": e.get("a_face"), "b_face": e.get("b_face")},
        "before_obb": other_obb,
        "after_obb": after_obb,
        "op": {
            "type": "scale_stub",
            "ratio": float(edit_decomp["r"]),
            "debug": scale_debug,
        },
    }


# ----------------------------
# Volume attachment: closest-face-to-volume-box then V1/V2/V3 (scaling delegated)
# ----------------------------

def _interval_to_plane_distance(lo: float, hi: float, p: float) -> float:
    """
    Distance from 1D interval [lo,hi] to plane coordinate p.
    If p is inside interval -> 0, else min endpoint distance.
    """
    lo = float(min(lo, hi))
    hi = float(max(lo, hi))
    if lo <= p <= hi:
        return 0.0
    return min(abs(lo - p), abs(hi - p))


def _closest_target_face_to_volume_box(
    bmin: np.ndarray,
    bmax: np.ndarray,
    E0: np.ndarray
) -> Tuple[int, int, float, Dict[str, Any]]:
    """
    In A-before local coords, target faces are planes:
      q[i] = +E0[i]  and  q[i] = -E0[i]
    Given a volume/contact box [bmin,bmax] (A-local), find the closest face plane.
    """
    best_axis, best_sign, best_dist = 0, +1, 1e30
    debug_faces: List[Dict[str, Any]] = []

    for i in range(3):
        lo = float(bmin[i])
        hi = float(bmax[i])

        p_plus = float(+E0[i])
        d_plus = _interval_to_plane_distance(lo, hi, p_plus)
        debug_faces.append({"axis": i, "sign": +1, "plane": p_plus, "interval": [min(lo, hi), max(lo, hi)], "dist": float(d_plus)})
        if d_plus < best_dist:
            best_axis, best_sign, best_dist = i, +1, float(d_plus)

        p_minus = float(-E0[i])
        d_minus = _interval_to_plane_distance(lo, hi, p_minus)
        debug_faces.append({"axis": i, "sign": -1, "plane": p_minus, "interval": [min(lo, hi), max(lo, hi)], "dist": float(d_minus)})
        if d_minus < best_dist:
            best_axis, best_sign, best_dist = i, -1, float(d_minus)

    debug = {
        "faces_checked": debug_faces,
        "chosen": {"axis": int(best_axis), "sign": int(best_sign), "dist": float(best_dist)},
    }
    return int(best_axis), int(best_sign), float(best_dist), debug


def _get_volume_box_local_from_edge(
    e: Dict[str, Any],
    edit_decomp: Dict[str, Any]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Dict[str, Any]]:
    """
    Try to extract a TARGET-LOCAL (A-before local) volume/contact box from edge.
    """
    bmin = e.get("overlap_box_local_min", None)
    bmax = e.get("overlap_box_local_max", None)
    if bmin is not None and bmax is not None:
        return _as_np(bmin), _as_np(bmax), "volume_box_local", {}

    aw = e.get("anchor_world", None)
    if aw is not None:
        P = _as_np(aw)
        q = _world_to_local(P, edit_decomp["C0"], edit_decomp["U0"])
        return q.copy(), q.copy(), "anchor_as_degenerate_box", {"anchor_world": P.tolist(), "anchor_local": q.tolist()}

    return None, None, "none", {}


def _classify_volume_attachment_face_vs_edit(
    e: Dict[str, Any],
    edit_decomp: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Determine where the VOLUME attachment is relative to the edited face by:
      - inferring which TARGET face (+/-u0/u1/u2) is closest to the contact volume box
      - comparing that face to the edited face (k, s_edit)
    """
    k = int(edit_decomp["axis"])
    s_edit = int(edit_decomp["s_edit"])
    E0 = edit_decomp["E0"]

    bmin, bmax, method, extra = _get_volume_box_local_from_edge(e, edit_decomp)
    if bmin is None or bmax is None:
        return "unknown", {
            "method": method,
            "reason": "no_volume_box_local_minmax_and_no_anchor",
            "edited_face": _face_to_str(k, s_edit),
        }

    axis_star, sign_star, dist_star, dbg = _closest_target_face_to_volume_box(bmin, bmax, E0)

    if axis_star == k and sign_star == s_edit:
        rel = "same"
    elif axis_star == k and sign_star == -s_edit:
        rel = "opposite"
    else:
        rel = "perpendicular"

    info = {
        "method": method,
        "edited_face": _face_to_str(k, s_edit),
        "closest_face": _face_to_str(axis_star, sign_star),
        "closest_face_axis": int(axis_star),
        "closest_face_sign": int(sign_star),
        "closest_face_dist": float(dist_star),
        "volume_box_local_min": bmin.tolist(),
        "volume_box_local_max": bmax.tolist(),
        "debug_faces": dbg.get("faces_checked", []),
    }
    info.update(extra)
    return rel, info


def _solve_volume_edge(
    target: str,
    other: str,
    e: Dict[str, Any],
    edit_decomp: Dict[str, Any],
    other_obb: Dict[str, Any],
    min_extent: float,
    constraints: Dict[str, Any],
    verbose: bool
) -> Optional[Dict[str, Any]]:
    k = int(edit_decomp["axis"])
    s_edit = int(edit_decomp["s_edit"])
    edited_face_str = _face_to_str(k, s_edit)

    rel, info = _classify_volume_attachment_face_vs_edit(e, edit_decomp)

    # SAME -> translate (V1)
    if rel == "same":
        solving = "V1(same_as_edited_face)->translate"
        where = f"volume_closest_face == edited_face({edited_face_str}) | closest={info.get('closest_face')} method={info.get('method')}"
        after_obb = _apply_translation(other_obb, edit_decomp["dF"])

        if verbose:
            _print_where_and_op(where=where, op="translation")
            print(f"[AEP][attach][VOLUME] solving={solving} | delta(dF)={edit_decomp['dF'].tolist()}")
            print(f"[AEP][attach][VOLUME]   closest_face={info.get('closest_face')} dist={info.get('closest_face_dist'):.6f} method={info.get('method')}")

        return {
            "kind": "volume",
            "solving": solving,
            "edge": {
                "a": e.get("a"),
                "b": e.get("b"),
                "anchor_world": e.get("anchor_world", None),
                "overlap_box_local_min": e.get("overlap_box_local_min", None),
                "overlap_box_local_max": e.get("overlap_box_local_max", None),
            },
            "before_obb": other_obb,
            "after_obb": after_obb,
            "op": {
                "type": "translate",
                "delta_world": edit_decomp["dF"].tolist(),
                "reason": "volume_same_face_as_edit",
                "debug": info,
            },
        }

    # OPPOSITE -> no change (V2)
    if rel == "opposite":
        solving = "V2(opposite_of_edited_face)->no_change"
        where = f"volume_closest_face is opposite_of edited_face({edited_face_str}) | closest={info.get('closest_face')} method={info.get('method')}"
        if verbose:
            _print_where_and_op(where=where, op="none")
            print(f"[AEP][attach][VOLUME] solving={solving} | no change")
            print(f"[AEP][attach][VOLUME]   closest_face={info.get('closest_face')} dist={info.get('closest_face_dist'):.6f} method={info.get('method')}")
        return None

    # PERPENDICULAR -> delegate scaling (V3)
    if rel == "perpendicular":
        solving = "V3(perpendicular_to_edited_face)->scale"
        where = f"volume_closest_face perpendicular_to edited_face({edited_face_str}) | closest={info.get('closest_face')} method={info.get('method')}"

        after_obb, scale_debug = scale_neighbor_obb(
            other_obb=other_obb,
            edit_face_normal=edit_decomp["n_edit"],
            scale_ratio=edit_decomp["r"],
            min_extent=min_extent,
            edited_face_str=edited_face_str,
            neighbor_name=other,
            verbose=verbose,
        )

        if verbose:
            _print_where_and_op(where=where, op="scaling")
            print(f"[AEP][attach][VOLUME] solving={solving} | delegated_scaling=stub (no actual change)")
            print(f"[AEP][attach][VOLUME]   closest_face={info.get('closest_face')} dist={info.get('closest_face_dist'):.6f} method={info.get('method')}")

        return {
            "kind": "volume",
            "solving": solving,
            "edge": {
                "a": e.get("a"),
                "b": e.get("b"),
                "anchor_world": e.get("anchor_world", None),
                "overlap_box_local_min": e.get("overlap_box_local_min", None),
                "overlap_box_local_max": e.get("overlap_box_local_max", None),
            },
            "before_obb": other_obb,
            "after_obb": after_obb,
            "op": {
                "type": "scale_stub",
                "ratio": float(edit_decomp["r"]),
                "reason": "volume_perpendicular_to_edit_face",
                "debug": {"volume_info": info, "scale_debug": scale_debug},
            },
        }

    # UNKNOWN -> conservative delegate scaling
    solving = "V?(unknown_volume_face)->scale"
    where = f"cannot_classify_volume_face_vs_edited_face({edited_face_str}) -> scale (conservative)"

    after_obb, scale_debug = scale_neighbor_obb(
        other_obb=other_obb,
        edit_face_normal=edit_decomp["n_edit"],
        scale_ratio=edit_decomp["r"],
        min_extent=min_extent,
        edited_face_str=edited_face_str,
        neighbor_name=other,
        verbose=verbose,
    )

    if verbose:
        _print_where_and_op(where=where, op="scaling")
        print(f"[AEP][attach][VOLUME] solving={solving} | delegated_scaling=stub (no actual change)")

    return {
        "kind": "volume",
        "solving": solving,
        "edge": {
            "a": e.get("a"),
            "b": e.get("b"),
            "anchor_world": e.get("anchor_world", None),
            "overlap_box_local_min": e.get("overlap_box_local_min", None),
            "overlap_box_local_max": e.get("overlap_box_local_max", None),
        },
        "before_obb": other_obb,
        "after_obb": after_obb,
        "op": {
            "type": "scale_stub",
            "ratio": float(edit_decomp["r"]),
            "reason": "unknown_volume_face_classification",
            "debug": {"volume_info": info, "scale_debug": scale_debug},
        },
    }


# ----------------------------
# Point attachment: move anchor (P1), then translate neighbor (P2) (UNCHANGED)
# ----------------------------

def _solve_point_edge(
    target: str,
    other: str,
    e: Dict[str, Any],
    edit_decomp: Dict[str, Any],
    other_obb: Dict[str, Any],
    verbose: bool
) -> Optional[Dict[str, Any]]:
    k = int(edit_decomp["axis"])
    s_edit = int(edit_decomp["s_edit"])
    edited_face_str = _face_to_str(k, s_edit)

    aw = e.get("anchor_world", None)
    if aw is None:
        solving = "P0(no_anchor)->skip"
        where = f"anchor_missing -> cannot_determine_closest_face_vs_edited_face({edited_face_str})"
        if verbose:
            _print_where_and_op(where=where, op="none")
            print(f"[AEP][attach][POINT] solving={solving} | skipped (no anchor_world)")
        return None

    P0 = _as_np(aw)

    # Determine closest face in A-before (for printing "where")
    q0 = _world_to_local(P0, edit_decomp["C0"], edit_decomp["U0"])
    i_star, s_star, delta = _closest_face_relation_local(q0, edit_decomp["E0"])
    closest_face_str = _face_to_str(i_star, s_star)
    if i_star == k and s_star == s_edit:
        where = f"anchor_closest_to_edited_face({edited_face_str}) (closest_face={closest_face_str}, delta={delta:.6f})"
    else:
        where = f"anchor_closest_to_face({closest_face_str}) NOT edited_face({edited_face_str}) (delta={delta:.6f})"

    # P1: move anchor so A-point relation unchanged
    P1, infoA = _move_anchor_preserve_A_relation(P0, edit_decomp)

    # P2: translate B rigidly by anchor displacement
    dP = P1 - P0
    after_obb = _apply_translation(other_obb, dP)

    if verbose:
        _print_where_and_op(where=where, op="translation")
        print(f"[AEP][attach][POINT] solving=P1+P2(rigid) | delta(dP)={dP.tolist()}")
        print(f"[AEP][attach][POINT]   P0={P0.tolist()}")
        print(f"[AEP][attach][POINT]   P1={P1.tolist()}")
        print(f"[AEP][attach][POINT]   preserved_face={_face_to_str(infoA['A_face_axis'], infoA['A_face_sign'])} offset(delta)={infoA['A_delta']:.6f}")

    return {
        "kind": "point",
        "solving": "P1+P2(rigid)",
        "edge": {"a": e.get("a"), "b": e.get("b"), "anchor_world": aw},
        "anchor": {
            "P0": P0.tolist(),
            "P1": P1.tolist(),
            "infoA": infoA,
        },
        "before_obb": other_obb,
        "after_obb": after_obb,
        "op": {"type": "translate", "delta_world": dP.tolist(), "reason": "preserve_point_relations"},
    }


# ----------------------------
# Public API (DO NOT BREAK)
# ----------------------------

def apply_attachments(constraints: Dict[str, Any], edit: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    attachments = constraints.get("attachments", []) or []

    target = edit.get("target", None)
    if not isinstance(target, str) or not target:
        raise ValueError("edit missing valid 'target'")

    change = edit.get("change", None)
    if not isinstance(change, dict):
        raise ValueError("edit missing 'change' dict")

    # edit decomposition
    ed = _compute_edit_decomp(change)
    min_extent = _min_extent_from_edit(change)

    # collect target-involved edges
    all_edges = [e for e in attachments if isinstance(e, dict)]
    target_edges = [e for e in all_edges if e.get("a") == target or e.get("b") == target]

    if verbose:
        counts_all = {"volume": 0, "face": 0, "point": 0, "unknown": 0}
        for e in all_edges:
            k = _infer_attachment_kind(e)
            if k not in counts_all:
                k = "unknown"
            counts_all[k] += 1
        print(f"[AEP][attach] ALL edges in file: {len(all_edges)}")
        print(f"[AEP][attach] ALL counts: volume={counts_all['volume']} face={counts_all['face']} point={counts_all['point']} unknown={counts_all['unknown']}")

        print(f"[AEP][attach] target={target} | edges_touching_target={len(target_edges)}")
        print(f"[AEP][attach] edited_face={_face_to_str(ed['axis'], ed['s_edit'])} | axis={ed['axis']} s_edit={ed['s_edit']} r={ed['r']:.6f}")
        print(f"[AEP][attach] dF(edited_face_disp)={ed['dF'].tolist()} | dC(center_disp)={ed['dC'].tolist()}")
        print(f"[AEP][attach] n_edit(outward_normal)={ed['n_edit'].tolist()}")

    changed_nodes: Dict[str, Any] = {}
    applied_any = False

    # Solve each target edge
    for idx, e in enumerate(target_edges):
        kind = _infer_attachment_kind(e)
        other = _other_name_in_edge(e, target)
        if not other:
            if verbose:
                print(f"\n[AEP][attach] edge#{idx} [SKIP] missing other endpoint: a={e.get('a')} b={e.get('b')}")
            continue

        other_obb = _get_node_obb(constraints, other)

        if verbose:
            _print_edge_header(kind=kind, target=target, other=other, idx=idx)

        rec = None
        if kind == "face":
            rec = _solve_face_edge(target, other, e, ed, other_obb, min_extent=min_extent, verbose=verbose)
        elif kind == "volume":
            rec = _solve_volume_edge(target, other, e, ed, other_obb, min_extent=min_extent, constraints=constraints, verbose=verbose)
        elif kind == "point":
            rec = _solve_point_edge(target, other, e, ed, other_obb, verbose=verbose)
        else:
            if verbose:
                _print_where_and_op(where="unknown_kind", op="none")
                print(f"[AEP][attach] [SKIP] unknown kind for edge#{idx}")

        if rec is None:
            continue

        # If multiple edges touch the same neighbor, we currently "last one wins".
        changed_nodes[other] = rec
        applied_any = True

        if verbose:
            aob = rec["after_obb"]
            print(f"[AEP][attach] RESULT saved for neighbor '{other}': kind={rec['kind']} solving={rec['solving']} op={rec['op'].get('type')}")
            print(f"[AEP][attach]   after_obb.center={aob['center']} extents={aob['extents']}")

    # build summary
    counts_target = {"volume": 0, "face": 0, "point": 0, "unknown": 0}
    for e in target_edges:
        k = _infer_attachment_kind(e)
        if k not in counts_target:
            k = "unknown"
        counts_target[k] += 1

    return {
        "target": target,
        "applied": bool(applied_any),
        "changed_nodes": changed_nodes,  # per-neighbor after_obb + logs
        "summary": {
            "total_edges": int(len(target_edges)),
            "counts": counts_target,
        },
    }
