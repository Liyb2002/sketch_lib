#!/usr/bin/env python3
# AEP/sym_and_containment.py

from typing import Dict, Any, List, Tuple
import numpy as np


# -----------------------------
# Face / axis parsing
# -----------------------------

def parse_face(face: str) -> Tuple[int, int]:
    if not isinstance(face, str) or len(face) != 3 or face[0] not in "+-" or face[1] != "u" or face[2] not in "012":
        raise ValueError(f"Invalid face '{face}'. Must be one of: +u0 -u0 +u1 -u1 +u2 -u2")
    s = +1 if face[0] == "+" else -1
    k = int(face[2])
    return k, s


def face_str(axis_k: int, sign_s: int) -> str:
    return ("+" if sign_s >= 0 else "-") + f"u{int(axis_k)}"


# -----------------------------
# Axis mapping between two OBB frames
# -----------------------------

def infer_axis_mapping(R_src: np.ndarray, R_dst: np.ndarray) -> Tuple[List[int], List[int], np.ndarray]:
    """
    Returns:
      perm[k] = j  (src axis k corresponds to dst axis j)
      sgn[k] = +1/-1  (dst_u_j ~= sgn[k] * src_u_k)
      M = R_dst^T R_src (dot matrix)
    """
    if R_src.shape != (3, 3) or R_dst.shape != (3, 3):
        raise ValueError("R_src and R_dst must be 3x3 with axes as columns.")

    M = R_dst.T @ R_src  # (3,3): rows=dst axes, cols=src axes

    perm = [-1, -1, -1]
    sgn = [0, 0, 0]
    used_dst = set()

    for k in range(3):
        best_j = None
        best_val = -1.0
        for j in range(3):
            if j in used_dst:
                continue
            v = abs(float(M[j, k]))
            if v > best_val:
                best_val = v
                best_j = j
        if best_j is None:
            raise RuntimeError("Failed to infer axis mapping.")

        used_dst.add(best_j)
        perm[k] = int(best_j)
        sgn[k] = +1 if float(M[best_j, k]) >= 0 else -1

    return perm, sgn, M


def map_face_from_src_to_dst(face_src: str, R_src: np.ndarray, R_dst: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    """
    Returns:
      face_dst, debug_info
    """
    k_src, s_src = parse_face(face_src)
    perm, sgn, M = infer_axis_mapping(R_src, R_dst)

    j_dst = perm[k_src]
    align_sign = sgn[k_src]
    s_dst = s_src * align_sign
    face_dst = face_str(j_dst, s_dst)

    dbg = {
        "M": M.tolist(),
        "perm_src_to_dst": perm,
        "sign_src_to_dst": sgn,
        "k_src": int(k_src),
        "j_dst": int(j_dst),
        "align_sign": int(align_sign),
        "s_src": int(s_src),
        "s_dst": int(s_dst),
        "face_src": face_src,
        "face_dst": face_dst,
    }
    return face_dst, dbg


# -----------------------------
# One-face move (opposite face fixed)
# -----------------------------

def apply_face_move_one_face_fixed(
    pre_obb: Dict[str, Any],
    face: str,
    delta: float,
    min_extent: float = 1e-4,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    k, s = parse_face(face)

    c = np.array(pre_obb["center"], dtype=np.float64)
    R = np.array(pre_obb["axes"], dtype=np.float64)      # columns
    e = np.array(pre_obb["extents"], dtype=np.float64)   # half extents

    old_e = float(e[k])
    u_k = R[:, k]

    # positions of the two faces along that axis in world:
    # +face plane point: c + e*u_k; -face: c - e*u_k
    plus_before = (c + old_e * u_k).copy()
    minus_before = (c - old_e * u_k).copy()

    desired_new_e = old_e + 0.5 * float(delta)
    if desired_new_e < float(min_extent):
        delta_applied = 2.0 * (float(min_extent) - old_e)
    else:
        delta_applied = float(delta)

    new_e_k = old_e + 0.5 * delta_applied
    if new_e_k < float(min_extent) - 1e-12:
        new_e_k = float(min_extent)

    c2 = c + (s * 0.5 * delta_applied) * u_k

    new_e = e.copy()
    new_e[k] = new_e_k

    plus_after = (c2 + new_e_k * u_k).copy()
    minus_after = (c2 - new_e_k * u_k).copy()

    post_obb = {
        "center": c2.tolist(),
        "axes": pre_obb["axes"],
        "extents": new_e.tolist(),
    }

    info = {
        "face": face,
        "axis": int(k),
        "sign": int(s),
        "delta_requested": float(delta),
        "delta_applied": float(delta_applied),
        "old_extent": float(old_e),
        "new_extent": float(new_e_k),
        "plus_face_before": plus_before.tolist(),
        "minus_face_before": minus_before.tolist(),
        "plus_face_after": plus_after.tolist(),
        "minus_face_after": minus_after.tolist(),
        "moved_face_expected": ("plus" if s > 0 else "minus"),
    }
    return post_obb, info


# -----------------------------
# Main API
# -----------------------------

def apply_symmetry_and_containment(
    constraints: Dict[str, Any],
    edit: Dict[str, Any],
    min_extent: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Applies proportional one-face edits to symmetry + containment neighbors.

    Prints verification logs:
    - relation
    - target edit details
    - axis mapping details
    - neighbor before/after summary
    """
    nodes = constraints.get("nodes", {}) or {}
    if not isinstance(nodes, dict) or len(nodes) == 0:
        raise ValueError("constraints['nodes'] missing/empty")

    target = edit.get("target", None)
    if not target:
        raise ValueError("edit missing 'target'")

    ch = edit.get("change", {}) or {}
    face_src = ch.get("face", None)
    if face_src is None:
        raise ValueError("edit['change'] missing 'face'")

    k_src, s_src = parse_face(face_src)

    if target not in nodes or nodes[target].get("obb", None) is None:
        raise ValueError(f"Target '{target}' missing from constraints['nodes'] or has no obb")

    target_obb_before = nodes[target]["obb"]
    R_src = np.array(target_obb_before["axes"], dtype=np.float64)
    e_src = float(target_obb_before["extents"][k_src])

    delta_src = ch.get("delta_applied", ch.get("delta_requested", None))
    if delta_src is None:
        raise ValueError("edit['change'] missing delta_applied/delta_requested")

    if e_src <= 0:
        raise ValueError(f"Target '{target}' has non-positive extent on axis u{k_src}: {e_src}")

    signed_ratio = float(delta_src) / float(e_src)

    symmetry = constraints.get("symmetry", {}) or {}
    containment = constraints.get("containment", []) or []

    # Gather relations involving target (so we can print them cleanly)
    sym_pairs_hit = []
    for p in symmetry.get("pairs", []) or []:
        a = p.get("a"); b = p.get("b")
        if a == target or b == target:
            sym_pairs_hit.append(p)

    cont_edges_hit = []
    for c in containment:
        if c.get("outer") == target or c.get("inner") == target:
            cont_edges_hit.append(c)

    if verbose:
        u_src = R_src[:, k_src].tolist()
        print("\n[AEP][SYM/CON][VERIFY] =======================================")
        print(f"[AEP][SYM/CON][VERIFY] target={target}")
        print(f"[AEP][SYM/CON][VERIFY] target_face={face_src} (axis=u{k_src}, sign={s_src})")
        print(f"[AEP][SYM/CON][VERIFY] target_axis_world(u{k_src})={u_src}")
        print(f"[AEP][SYM/CON][VERIFY] target_extent(u{k_src})={e_src:.6f}")
        print(f"[AEP][SYM/CON][VERIFY] delta_src(applied)={float(delta_src):+.6f}")
        print(f"[AEP][SYM/CON][VERIFY] signed_ratio=delta_src/extent={signed_ratio:+.6f}")
        print(f"[AEP][SYM/CON][VERIFY] symmetry_edges_hit={len(sym_pairs_hit)} containment_edges_hit={len(cont_edges_hit)}")
        print("[AEP][SYM/CON][VERIFY] =======================================\n")

    out_sym: Dict[str, Any] = {}
    out_con: Dict[str, Any] = {}

    def _apply_to_neighbor(neighbor: str, rel_tag: str, rel_obj: Dict[str, Any]) -> Dict[str, Any]:
        if neighbor not in nodes or nodes[neighbor].get("obb", None) is None:
            print(f"[AEP][SYM/CON][VERIFY][WARN] neighbor '{neighbor}' missing obb. skip.")
            return {}

        nb_before = nodes[neighbor]["obb"]
        R_dst = np.array(nb_before["axes"], dtype=np.float64)

        face_dst, map_dbg = map_face_from_src_to_dst(face_src, R_src=R_src, R_dst=R_dst)
        k_dst, s_dst = parse_face(face_dst)
        e_dst = float(nb_before["extents"][k_dst])

        delta_dst_req = signed_ratio * e_dst

        nb_after, info = apply_face_move_one_face_fixed(
            pre_obb=nb_before,
            face=face_dst,
            delta=float(delta_dst_req),
            min_extent=float(min_extent),
        )


        # sanity: show that only ONE face moved (the opposite should remain unchanged)
        # Determine which face should be fixed:
        if s_dst > 0:
            # + face moved, - face fixed
            fixed_before = np.array(info["minus_face_before"])
            fixed_after = np.array(info["minus_face_after"])
            moved_before = np.array(info["plus_face_before"])
            moved_after = np.array(info["plus_face_after"])
            fixed_name = "-face"
            moved_name = "+face"
        else:
            fixed_before = np.array(info["plus_face_before"])
            fixed_after = np.array(info["plus_face_after"])
            moved_before = np.array(info["minus_face_before"])
            moved_after = np.array(info["minus_face_after"])
            fixed_name = "+face"
            moved_name = "-face"


        return {
            "relation_tag": rel_tag,
            "relation": rel_obj,
            "mapped_face": face_dst,
            "signed_ratio": float(signed_ratio),
            "delta_dst_requested": float(delta_dst_req),
            "delta_dst_applied": float(info["delta_applied"]),
            "before_obb": nb_before,
            "after_obb": nb_after,
            "mapping_debug": map_dbg,
            "face_motion_debug": {
                "plus_before": info["plus_face_before"],
                "plus_after": info["plus_face_after"],
                "minus_before": info["minus_face_before"],
                "minus_after": info["minus_face_after"],
            },
        }

    # --- symmetry: apply per pair
    for p in sym_pairs_hit:
        a = p.get("a"); b = p.get("b")
        other = b if a == target else a
        if other in out_sym or other in out_con:
            continue
        r = _apply_to_neighbor(other, "SYM", p)
        if r:
            out_sym[other] = r

    # --- containment: apply per edge
    for c in cont_edges_hit:
        outer = c.get("outer"); inner = c.get("inner")
        other = inner if outer == target else outer
        if other in out_sym or other in out_con:
            continue
        r = _apply_to_neighbor(other, "CONTAIN", c)
        if r:
            out_con[other] = r

    return {"symmetry": out_sym, "containment": out_con}
