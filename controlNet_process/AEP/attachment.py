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

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


# ----------------------------
# Basic helpers
# ----------------------------

def _interval_to_plane_distance(a: float, b: float, p: float) -> float:
    # distance from 1D interval [a,b] to point p
    lo = float(min(a, b))
    hi = float(max(a, b))
    if lo <= p <= hi:
        return 0.0
    return min(abs(lo - p), abs(hi - p))


def _volume_where_vs_edited_face(
    e: Dict[str, Any],
    edit_decomp: Dict[str, Any],
    target: str,
    verbose: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns:
      where_str: one of
        - "closer_to_edited_face"
        - "closer_to_opposite_face"
        - "tie_choose_edited"
        - "no_volume_box_fallback_anchor"
        - "no_volume_box_no_anchor"
      info: debug dict
    """
    k = int(edit_decomp["axis"])
    s_edit = int(edit_decomp["s_edit"])
    E0 = edit_decomp["E0"]
    p_edit = float(s_edit * E0[k])
    p_opp  = float(-s_edit * E0[k])

    # Preferred: use overlap/contact box in TARGET-LOCAL coords if available
    # (Your pipeline may store these; adjust key names if yours differ.)
    bmin = e.get("overlap_box_local_min", None)
    bmax = e.get("overlap_box_local_max", None)

    if bmin is not None and bmax is not None:
        bmin = _as_np(bmin)
        bmax = _as_np(bmax)
        a = float(bmin[k])
        b = float(bmax[k])

        d_edit = _interval_to_plane_distance(a, b, p_edit)
        d_opp  = _interval_to_plane_distance(a, b, p_opp)

        if d_edit < d_opp:
            where = "closer_to_edited_face"
        elif d_opp < d_edit:
            where = "closer_to_opposite_face"
        else:
            where = "tie_choose_edited"

        info = {
            "method": "volume_box_local",
            "axis": k,
            "p_edit": p_edit,
            "p_opp": p_opp,
            "interval_k": [min(a, b), max(a, b)],
            "dist_edit": d_edit,
            "dist_opp": d_opp,
        }
        return where, info

    # Fallback: if you still have anchor_world, compare its distances to the two planes
    aw = e.get("anchor_world", None)
    if aw is not None:
        P = _as_np(aw)
        q = _world_to_local(P, edit_decomp["C0"], edit_decomp["U0"])
        qk = float(q[k])

        d_edit = abs(qk - p_edit)
        d_opp  = abs(qk - p_opp)

        if d_edit < d_opp:
            where = "closer_to_edited_face"
        elif d_opp < d_edit:
            where = "closer_to_opposite_face"
        else:
            where = "tie_choose_edited"

        info = {
            "method": "anchor_fallback",
            "axis": k,
            "p_edit": p_edit,
            "p_opp": p_opp,
            "qk": qk,
            "dist_edit": d_edit,
            "dist_opp": d_opp,
        }
        return where, info

    return "no_volume_box_no_anchor", {"method": "none"}



def _as_np(x):
    return np.asarray(x, dtype=np.float64)


def _safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(v) + eps)


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


def _get_tols(constraints: Dict[str, Any], E0k: float) -> Tuple[float, float]:
    params = constraints.get("params", {}) or {}
    gap = params.get("attach_face_gap_tol", 0.01)
    ov = params.get("attach_face_overlap_tol", 0.01)
    try:
        gap = float(gap)
    except Exception:
        gap = 0.01
    try:
        ov = float(ov)
    except Exception:
        ov = 0.01

    # also allow relative fallback
    gap = max(gap, 0.05 * float(E0k))
    ov = max(ov, 0.05 * float(E0k))
    return gap, ov


# ----------------------------
# Geometry primitives for AEP rules
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
        # fallback: use sign in json if present
        s_edit = int(change.get("sign", +1))
        s_edit = +1 if s_edit >= 0 else -1
    else:
        k_face, s_edit = parsed
        if axis in (0, 1, 2) and k_face != axis:
            # axis mismatch in file; trust "axis"
            pass
    if axis not in (0, 1, 2):
        raise ValueError(f"Invalid edit axis: {axis}")

    # scale ratio along edited axis
    r = float(E1[axis] / max(E0[axis], 1e-12))

    # face displacement for edited face
    F0 = _face_center(C0, U0, E0, axis, s_edit)
    F1 = _face_center(C1, U1, E1, axis, s_edit)
    dF = F1 - F0

    # also keep center translation and axis direction
    dC = C1 - C0
    uk = _normalize(U0[axis])

    return {
        "C0": C0, "U0": U0, "E0": E0,
        "C1": C1, "U1": U1, "E1": E1,
        "axis": axis,
        "s_edit": int(s_edit),
        "r": float(r),
        "dF": dF,
        "dC": dC,
        "uk": uk,
    }


def _choose_neighbor_axis_aligned_with_dir(U_rows_neighbor: np.ndarray, d_world: np.ndarray) -> int:
    d = _normalize(d_world)
    dots = [abs(float(np.dot(U_rows_neighbor[i], d))) for i in range(3)]
    return int(np.argmax(dots))


def _apply_scale_along_dir(obb: Dict[str, Any], d_world: np.ndarray, r: float, min_extent: float) -> Tuple[Dict[str, Any], int, float, float]:
    C = _as_np(obb["center"])
    U = _axes_matrix(obb["axes"])
    E = _as_np(obb["extents"]).copy()

    m = _choose_neighbor_axis_aligned_with_dir(U, d_world)
    old = float(E[m])
    new = float(max(min_extent, old * float(r)))
    E[m] = new

    out = {
        "center": C.tolist(),
        "axes": U.tolist(),
        "extents": E.tolist(),
    }
    return out, m, old, new


def _apply_translation(obb: Dict[str, Any], d_world: np.ndarray) -> Dict[str, Any]:
    C = _as_np(obb["center"])
    U = _axes_matrix(obb["axes"])
    E = _as_np(obb["extents"])
    C2 = C + d_world
    return {"center": C2.tolist(), "axes": U.tolist(), "extents": E.tolist()}


def _anchor_on_edited_face(anchor_world: np.ndarray, edit_decomp: Dict[str, Any], gap_tol: float, in_tol: float) -> bool:
    # Use BEFORE box for classification
    C0, U0, E0 = edit_decomp["C0"], edit_decomp["U0"], edit_decomp["E0"]
    k, s = edit_decomp["axis"], edit_decomp["s_edit"]

    q = _world_to_local(anchor_world, C0, U0)
    # near plane
    if abs(float(q[k] - s * E0[k])) > float(gap_tol):
        return False
    # inside face rectangle (with tolerance)
    for i in (0, 1, 2):
        if i == k:
            continue
        if abs(float(q[i])) > float(E0[i] + in_tol):
            return False
    return True


# ----------------------------
# Point attachment helpers
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
    Compute new anchor P1 such that A-point relation is preserved from A-before to A-after.
    We preserve:
      - closest face id (axis + sign)
      - offset delta to that face plane
      - tangential local coords (q other axes) exactly
    """
    C0, U0, E0 = edit_decomp["C0"], edit_decomp["U0"], edit_decomp["E0"]
    C1, U1, E1 = edit_decomp["C1"], edit_decomp["U1"], edit_decomp["E1"]

    q0 = _world_to_local(P0, C0, U0)
    i_star, s_star, delta = _closest_face_relation_local(q0, E0)

    q1 = q0.copy()
    q1[i_star] = float(s_star * E1[i_star] + delta)  # preserve offset to same face plane

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
# Pretty-print helpers: exactly what you asked
# ----------------------------

def _print_edge_header(kind: str, target: str, other: str, idx: int):
    print(f"\n[AEP][attach] edge#{idx}  A(target)={target}  B(neighbor)={other}")
    print(f"[AEP][attach] 1) attachment_type = {kind}")


def _print_where_and_op(where: str, op: str):
    # op must be: "translation" | "scaling" | "none"
    print(f"[AEP][attach] 2) where_vs_edited_face = {where}")
    print(f"[AEP][attach] 3) operation = {op}")


# ----------------------------
# Face attachment: A1/A2/A3
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
        # can't classify; fall back to A3 behavior (scale) conservatively
        solving = "A3(face_fallback_no_face)"
        where = f"unknown_target_face -> treat_as_perpendicular_to_edited_face({edited_face_str})"
        after_obb, m, old, new = _apply_scale_along_dir(other_obb, edit_decomp["uk"], edit_decomp["r"], min_extent)
        if verbose:
            _print_where_and_op(where=where, op="scaling")
            print(f"[AEP][attach][FACE] solving={solving} | chosen_B_axis=u{m} | E: {old:.6f}->{new:.6f} | r={edit_decomp['r']:.6f}")
        return {
            "kind": "face",
            "solving": solving,
            "edge": {"a": e.get("a"), "b": e.get("b"), "a_face": e.get("a_face"), "b_face": e.get("b_face")},
            "before_obb": other_obb,
            "after_obb": after_obb,
            "op": {"type": "scale", "ratio": float(edit_decomp["r"]), "edit_dir": edit_decomp["uk"].tolist(), "axis_chosen": int(m)},
        }

    t_axis, t_sign = t_parsed

    if t_axis == k and t_sign == s_edit:
        # A1: same edited face
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
        # A2: opposite face
        solving = "A2"
        where = f"attached_on_opposite_face: target_face={_face_to_str(t_axis, t_sign)} opposite_of edited_face={edited_face_str}"
        if verbose:
            _print_where_and_op(where=where, op="none")
            print(f"[AEP][attach][FACE] solving={solving} | no change")
        return None

    # A3: perpendicular face
    solving = "A3"
    where = f"attached_on_perpendicular_face: target_face={_face_to_str(t_axis, t_sign)} perp_to edited_face={edited_face_str}"
    after_obb, m, old, new = _apply_scale_along_dir(other_obb, edit_decomp["uk"], edit_decomp["r"], min_extent)
    if verbose:
        _print_where_and_op(where=where, op="scaling")
        print(f"[AEP][attach][FACE] solving={solving} | chosen_B_axis=u{m} | E: {old:.6f}->{new:.6f} | r={edit_decomp['r']:.6f}")
    return {
        "kind": "face",
        "solving": solving,
        "edge": {"a": e.get("a"), "b": e.get("b"), "a_face": e.get("a_face"), "b_face": e.get("b_face")},
        "before_obb": other_obb,
        "after_obb": after_obb,
        "op": {"type": "scale", "ratio": float(edit_decomp["r"]), "edit_dir": edit_decomp["uk"].tolist(), "axis_chosen": int(m)},
    }


# ----------------------------
# Volume attachment: B1 => translate, else => scale
# ----------------------------

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
    """
    New volume rule (stable, no hard tolerance tuning):

    Decide whether the volume attachment is "on the edited face" by comparing
    distance to the edited face plane vs distance to the opposite face plane,
    using a volume box in TARGET-LOCAL coordinates when available.

    Priority:
      1) Use overlap/contact box in target-local coords:
           e["overlap_box_local_min"], e["overlap_box_local_max"]
         Compare distance from that interval (along edited axis k) to:
           p_edit = s_edit * E0[k]
           p_opp  = -s_edit * E0[k]
         If dist_edit <= dist_opp  => treat as "on edited face"  (translate)
         Else                       => treat as "on opposite side" (scale)

      2) Fallback to anchor_world if no local box:
         compute qk in A-before local coords and compare distances to p_edit/p_opp

      3) If neither exists: treat as opposite side => scale (conservative)
    """

    def _interval_to_plane_distance(a: float, b: float, p: float) -> float:
        lo = float(min(a, b))
        hi = float(max(a, b))
        if lo <= p <= hi:
            return 0.0
        return min(abs(lo - p), abs(hi - p))

    k = int(edit_decomp["axis"])
    s_edit = int(edit_decomp["s_edit"])
    edited_face_str = _face_to_str(k, s_edit)

    # face planes in A-before local coordinates
    E0 = edit_decomp["E0"]
    p_edit = float(s_edit * E0[k])
    p_opp = float(-s_edit * E0[k])

    # -----------------------------------------
    # Preferred: use volume overlap/contact box
    # -----------------------------------------
    bmin = e.get("overlap_box_local_min", None)
    bmax = e.get("overlap_box_local_max", None)

    where = None
    info = {"method": None}

    if bmin is not None and bmax is not None:
        bmin = _as_np(bmin)
        bmax = _as_np(bmax)
        a = float(bmin[k])
        b = float(bmax[k])

        d_edit = _interval_to_plane_distance(a, b, p_edit)
        d_opp = _interval_to_plane_distance(a, b, p_opp)

        if d_edit < d_opp:
            where = "closer_to_edited_face"
        elif d_opp < d_edit:
            where = "closer_to_opposite_face"
        else:
            where = "tie_choose_edited"

        info = {
            "method": "volume_box_local",
            "axis": k,
            "edited_face": edited_face_str,
            "p_edit": p_edit,
            "p_opp": p_opp,
            "interval_k": [min(a, b), max(a, b)],
            "dist_edit": d_edit,
            "dist_opp": d_opp,
        }

    # -----------------------------------------
    # Fallback: use anchor_world (relative test)
    # -----------------------------------------
    if where is None:
        aw = e.get("anchor_world", None)
        if aw is not None:
            P0 = _as_np(aw)
            q = _world_to_local(P0, edit_decomp["C0"], edit_decomp["U0"])
            qk = float(q[k])

            d_edit = abs(qk - p_edit)
            d_opp = abs(qk - p_opp)

            if d_edit < d_opp:
                where = "closer_to_edited_face"
            elif d_opp < d_edit:
                where = "closer_to_opposite_face"
            else:
                where = "tie_choose_edited"

            info = {
                "method": "anchor_local_k_only",
                "axis": k,
                "edited_face": edited_face_str,
                "p_edit": p_edit,
                "p_opp": p_opp,
                "qk": qk,
                "dist_edit": d_edit,
                "dist_opp": d_opp,
            }
        else:
            # nothing to classify => conservative scale
            where = "no_volume_box_no_anchor"
            info = {
                "method": "none",
                "axis": k,
                "edited_face": edited_face_str,
                "p_edit": p_edit,
                "p_opp": p_opp,
            }

    # -----------------------------------------
    # Decide operation
    # -----------------------------------------
    on_edited = (where in ("closer_to_edited_face", "tie_choose_edited"))

    if on_edited:
        # translate (same as "B1")
        solving = "B1(closer_to_edited)->translate"
        after_obb = _apply_translation(other_obb, edit_decomp["dF"])

        if verbose:
            _print_where_and_op(
                where=f"{where} | edited_face={edited_face_str} | method={info.get('method')} "
                      f"| dist_edit={info.get('dist_edit', None)} dist_opp={info.get('dist_opp', None)}",
                op="translation",
            )
            if info.get("method") == "volume_box_local":
                print(f"[AEP][attach][VOLUME] interval_k={info['interval_k']} p_edit={info['p_edit']:.6f} p_opp={info['p_opp']:.6f}")
            elif info.get("method") == "anchor_local_k_only":
                print(f"[AEP][attach][VOLUME] qk={info['qk']:.6f} p_edit={info['p_edit']:.6f} p_opp={info['p_opp']:.6f}")
            else:
                print(f"[AEP][attach][VOLUME] WARNING: no box/anchor, defaulted to translate due to tie policy (should be rare).")
            print(f"[AEP][attach][VOLUME] solving={solving} | delta(dF)={edit_decomp['dF'].tolist()}")

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
                "reason": "volume_closer_to_edited_face",
                "where": where,
                "debug": info,
            },
        }

    # else: closer to opposite => scale (same as A3)
    solving = "B2B3(closer_to_opposite)->scale"
    after_obb, m, old, new = _apply_scale_along_dir(other_obb, edit_decomp["uk"], edit_decomp["r"], min_extent)

    if verbose:
        _print_where_and_op(
            where=f"{where} | edited_face={edited_face_str} | method={info.get('method')} "
                  f"| dist_edit={info.get('dist_edit', None)} dist_opp={info.get('dist_opp', None)}",
            op="scaling",
        )
        if info.get("method") == "volume_box_local":
            print(f"[AEP][attach][VOLUME] interval_k={info['interval_k']} p_edit={info['p_edit']:.6f} p_opp={info['p_opp']:.6f}")
        elif info.get("method") == "anchor_local_k_only":
            print(f"[AEP][attach][VOLUME] qk={info['qk']:.6f} p_edit={info['p_edit']:.6f} p_opp={info['p_opp']:.6f}")
        else:
            print(f"[AEP][attach][VOLUME] WARNING: no box/anchor, defaulted to scale.")
        print(f"[AEP][attach][VOLUME] solving={solving} | chosen_B_axis=u{m} | E: {old:.6f}->{new:.6f} | r={edit_decomp['r']:.6f}")

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
            "type": "scale",
            "ratio": float(edit_decomp["r"]),
            "edit_dir": edit_decomp["uk"].tolist(),
            "axis_chosen": int(m),
            "reason": "volume_closer_to_opposite_face",
            "where": where,
            "debug": info,
        },
    }


# ----------------------------
# Point attachment: always translate (P1+P2)
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
        solving = "P(no_anchor)->skip"
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
