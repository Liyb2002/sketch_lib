#!/usr/bin/env python3
# AEP/attachment.py
#
# Computes attachment-driven AEP propagation (per your finalized rules),
# and prints:
#   - attachment kind (face / volume / point)
#   - solving type (A1/A2/A3, B1/B2, P1/P2, ...)
#   - output results (neighbor after_obb / translation / scaling / anchor move)
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
# Behavior summary (your decisions):
#   FACE:
#     A1 (same face): translate B by edited face displacement ΔF
#     A2 (opposite):  no change
#     A3 (perp):      scale B along edit direction by ratio r
#
#   VOLUME:
#     B1 (overlap touches edited face): same as A1 (translate by ΔF)
#     else (B2/B3):                    same as A3 (scale by r)
#
#   POINT (new rule):
#     Preserve relation "box <-> point" by:
#       P1: move anchor to preserve A-point relation (closest face + offset + tangential coords)
#       P2: translate B so that B-point relation is unchanged (rigid: ΔB = P1 - P0)
#     NOTE: this may be strong; if needed later, add damping or normal-only mode.

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


# ----------------------------
# Basic helpers
# ----------------------------

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
# Point attachment: preserve box<->point relation
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
# Face attachment: A1/A2/A3
# ----------------------------

def _solve_face_edge(target: str, other: str, e: Dict[str, Any], edit_decomp: Dict[str, Any], other_obb: Dict[str, Any],
                     min_extent: float, verbose: bool) -> Optional[Dict[str, Any]]:
    """
    Returns a change record for 'other' or None.
    """
    # Determine which face string corresponds to the target in this edge
    # If edge stores a_face/b_face, assume:
    #   if e['a']==target -> target_face=e['a_face'], other_face=e['b_face']
    #   else -> target_face=e['b_face'], other_face=e['a_face']
    if e.get("a") == target:
        t_face = e.get("a_face", None)
        o_face = e.get("b_face", None)
    else:
        t_face = e.get("b_face", None)
        o_face = e.get("a_face", None)

    t_parsed = _parse_face_str(t_face)
    e_parsed = (edit_decomp["axis"], edit_decomp["s_edit"])
    if t_parsed is None:
        # can't classify; fall back to A3 behavior (scale) conservatively
        solving = "A3(face_fallback_no_face)"
        after_obb, m, old, new = _apply_scale_along_dir(other_obb, edit_decomp["uk"], edit_decomp["r"], min_extent)
        if verbose:
            print(f"[AEP][attach][FACE] {target} <-> {other} | solving={solving} | scale_axis={m} old={old:.6f} new={new:.6f}")
        return {
            "kind": "face",
            "solving": solving,
            "edge": {"a": e.get("a"), "b": e.get("b"), "a_face": e.get("a_face"), "b_face": e.get("b_face")},
            "before_obb": other_obb,
            "after_obb": after_obb,
            "op": {"type": "scale", "ratio": float(edit_decomp["r"]), "edit_dir": edit_decomp["uk"].tolist(), "axis_chosen": int(m)},
        }

    t_axis, t_sign = t_parsed
    k, s_edit = e_parsed

    if t_axis == k and t_sign == s_edit:
        # A1
        solving = "A1"
        after_obb = _apply_translation(other_obb, edit_decomp["dF"])
        if verbose:
            print(f"[AEP][attach][FACE] {target} <-> {other} | solving={solving} | dF={edit_decomp['dF'].tolist()}")
            print(f"[AEP][attach][FACE]   other_center: {other_obb['center']} -> {after_obb['center']}")
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
        if verbose:
            print(f"[AEP][attach][FACE] {target} <-> {other} | solving={solving} | no change")
        return None

    # perpendicular -> A3
    solving = "A3"
    after_obb, m, old, new = _apply_scale_along_dir(other_obb, edit_decomp["uk"], edit_decomp["r"], min_extent)
    if verbose:
        print(f"[AEP][attach][FACE] {target} <-> {other} | solving={solving} | scale_axis={m} old={old:.6f} new={new:.6f} | r={edit_decomp['r']:.6f}")
    return {
        "kind": "face",
        "solving": solving,
        "edge": {"a": e.get("a"), "b": e.get("b"), "a_face": e.get("a_face"), "b_face": e.get("b_face")},
        "before_obb": other_obb,
        "after_obb": after_obb,
        "op": {"type": "scale", "ratio": float(edit_decomp["r"]), "edit_dir": edit_decomp["uk"].tolist(), "axis_chosen": int(m)},
    }


# ----------------------------
# Volume attachment: B1 else (B2/B3) => scale
# ----------------------------

def _solve_volume_edge(target: str, other: str, e: Dict[str, Any], edit_decomp: Dict[str, Any], other_obb: Dict[str, Any],
                       min_extent: float, constraints: Dict[str, Any], verbose: bool) -> Optional[Dict[str, Any]]:
    aw = e.get("anchor_world", None)
    if aw is None:
        # without anchor, treat as not-B1 => scale (A3)
        solving = "B2B3(no_anchor)->A3"
        after_obb, m, old, new = _apply_scale_along_dir(other_obb, edit_decomp["uk"], edit_decomp["r"], min_extent)
        if verbose:
            print(f"[AEP][attach][VOLUME] {target} <-> {other} | solving={solving} | scale_axis={m} old={old:.6f} new={new:.6f}")
        return {
            "kind": "volume",
            "solving": solving,
            "edge": {"a": e.get("a"), "b": e.get("b"), "anchor_world": None},
            "before_obb": other_obb,
            "after_obb": after_obb,
            "op": {"type": "scale", "ratio": float(edit_decomp["r"]), "edit_dir": edit_decomp["uk"].tolist(), "axis_chosen": int(m)},
        }

    P0 = _as_np(aw)
    gap_tol, in_tol = _get_tols(constraints, float(edit_decomp["E0"][edit_decomp["axis"]]))
    touches = _anchor_on_edited_face(P0, edit_decomp, gap_tol=gap_tol, in_tol=in_tol)

    if touches:
        # B1 => same as A1
        solving = "B1->A1"
        after_obb = _apply_translation(other_obb, edit_decomp["dF"])
        if verbose:
            print(f"[AEP][attach][VOLUME] {target} <-> {other} | solving={solving} | anchor_on_face=True | dF={edit_decomp['dF'].tolist()}")
            print(f"[AEP][attach][VOLUME]   other_center: {other_obb['center']} -> {after_obb['center']}")
        return {
            "kind": "volume",
            "solving": solving,
            "edge": {"a": e.get("a"), "b": e.get("b"), "anchor_world": aw},
            "before_obb": other_obb,
            "after_obb": after_obb,
            "op": {"type": "translate", "delta_world": edit_decomp["dF"].tolist(), "reason": "overlap_touches_edited_face"},
        }

    # else => A3 scaling
    solving = "B2B3->A3"
    after_obb, m, old, new = _apply_scale_along_dir(other_obb, edit_decomp["uk"], edit_decomp["r"], min_extent)
    if verbose:
        print(f"[AEP][attach][VOLUME] {target} <-> {other} | solving={solving} | anchor_on_face=False | scale_axis={m} old={old:.6f} new={new:.6f} | r={edit_decomp['r']:.6f}")
    return {
        "kind": "volume",
        "solving": solving,
        "edge": {"a": e.get("a"), "b": e.get("b"), "anchor_world": aw},
        "before_obb": other_obb,
        "after_obb": after_obb,
        "op": {"type": "scale", "ratio": float(edit_decomp["r"]), "edit_dir": edit_decomp["uk"].tolist(), "axis_chosen": int(m)},
    }


# ----------------------------
# Point attachment: move anchor (P1), then translate neighbor (P2)
# ----------------------------

def _solve_point_edge(target: str, other: str, e: Dict[str, Any], edit_decomp: Dict[str, Any], other_obb: Dict[str, Any],
                      verbose: bool) -> Optional[Dict[str, Any]]:
    aw = e.get("anchor_world", None)
    if aw is None:
        # nothing actionable
        solving = "P0(no_anchor)->skip"
        if verbose:
            print(f"[AEP][attach][POINT] {target} <-> {other} | solving={solving} | no anchor_world")
        return None

    P0 = _as_np(aw)

    # P1: move anchor so A-point relation unchanged
    P1, infoA = _move_anchor_preserve_A_relation(P0, edit_decomp)

    # P2: translate B rigidly by anchor displacement
    dP = P1 - P0
    after_obb = _apply_translation(other_obb, dP)

    if verbose:
        print(f"[AEP][attach][POINT] {target} <-> {other} | solving=P1+P2(rigid)")
        print(f"[AEP][attach][POINT]   P0={P0.tolist()}")
        print(f"[AEP][attach][POINT]   P1={P1.tolist()} | dP={dP.tolist()}")
        print(f"[AEP][attach][POINT]   A_face=(axis={infoA['A_face_axis']}, sign={infoA['A_face_sign']}) delta={infoA['A_delta']:.6f}")
        print(f"[AEP][attach][POINT]   other_center: {other_obb['center']} -> {after_obb['center']}")

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
# Public API
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

    # collect target-involved edges and also print all edges types
    all_edges = [e for e in attachments if isinstance(e, dict)]
    target_edges = [e for e in all_edges if e.get("a") == target or e.get("b") == target]

    if verbose:
        # Print global kinds
        counts_all = {"volume": 0, "face": 0, "point": 0, "unknown": 0}
        for e in all_edges:
            k = _infer_attachment_kind(e)
            if k not in counts_all:
                k = "unknown"
            counts_all[k] += 1
        print(f"[AEP][attach] ALL edges in file: {len(all_edges)}")
        print(f"[AEP][attach] ALL counts: volume={counts_all['volume']} face={counts_all['face']} point={counts_all['point']} unknown={counts_all['unknown']}")

        print(f"[AEP][attach] target={target} | edges_touching_target={len(target_edges)}")
        print(f"[AEP][attach] edit axis={ed['axis']} s_edit={ed['s_edit']} r={ed['r']:.6f}")
        print(f"[AEP][attach] dF(edited_face_disp)={ed['dF'].tolist()} | dC(center_disp)={ed['dC'].tolist()}")

    changed_nodes: Dict[str, Any] = {}
    applied_any = False

    # Solve each target edge
    for idx, e in enumerate(target_edges):
        kind = _infer_attachment_kind(e)
        other = _other_name_in_edge(e, target)
        if not other:
            if verbose:
                print(f"[AEP][attach]   [skip] edge#{idx} missing other endpoint: a={e.get('a')} b={e.get('b')}")
            continue

        other_obb = _get_node_obb(constraints, other)

        if verbose:
            print(f"\n[AEP][attach] --- edge#{idx} kind={kind} target={target} other={other} ---")

        rec = None
        if kind == "face":
            rec = _solve_face_edge(target, other, e, ed, other_obb, min_extent=min_extent, verbose=verbose)
        elif kind == "volume":
            rec = _solve_volume_edge(target, other, e, ed, other_obb, min_extent=min_extent, constraints=constraints, verbose=verbose)
        elif kind == "point":
            rec = _solve_point_edge(target, other, e, ed, other_obb, verbose=verbose)
        else:
            if verbose:
                print(f"[AEP][attach]   [skip] unknown kind for edge#{idx}")

        if rec is None:
            continue

        # If multiple edges touch the same neighbor, we currently "last one wins".
        # (Later: merge rules / accumulate translations, etc.)
        changed_nodes[other] = rec
        applied_any = True

        if verbose:
            aob = rec["after_obb"]
            print(f"[AEP][attach]   RESULT: other={other} | solving={rec['solving']}")
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
