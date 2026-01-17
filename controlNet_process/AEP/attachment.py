#!/usr/bin/env python3
# AEP/attachment.py
#
# Attachment propagation:
# For each neighbor N attached to target T:
#   Let F_edit be the edited face on target.
#   Let T_attach_face be the attachment face on T from the relation.
#
# Case A: F_edit == T_attach_face
#   -> translate neighbor along contact normal by target delta (same sign/magnitude)
#
# Case B: F_edit == opposite(T_attach_face)
#   -> do nothing
#
# Case C: orthogonal axes
#   -> scale neighbor along axis corresponding to target edited axis with same ratio r,
#      pivoted around anchor_world so the anchor point stays fixed in world.
#      No translation along contact normal.
#
# Returns a dict of changed neighbors:
#   { neighbor_name: {reason, case, mapped_axis, before_obb, after_obb, ...}, ... }

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


# -----------------------------
# Face helpers
# -----------------------------

def parse_face(face: str) -> Tuple[int, int]:
    if not isinstance(face, str) or len(face) != 3 or face[0] not in "+-" or face[1] != "u" or face[2] not in "012":
        raise ValueError(f"Invalid face '{face}'. Must be one of: +u0 -u0 +u1 -u1 +u2 -u2")
    s = +1 if face[0] == "+" else -1
    k = int(face[2])
    return k, s


def opposite_face(face: str) -> str:
    k, s = parse_face(face)
    return ("-" if s > 0 else "+") + f"u{k}"


def face_str(k: int, s: int) -> str:
    return ("+" if s >= 0 else "-") + f"u{int(k)}"


# -----------------------------
# Axis mapping (same as sym/contain approach)
# -----------------------------

def infer_axis_mapping(R_src: np.ndarray, R_dst: np.ndarray) -> Tuple[List[int], List[int], np.ndarray]:
    """
    Given 3x3 axes matrices (columns are axes), infer mapping from src axes -> dst axes.
    Returns:
      perm[k] = j  (src axis k corresponds to dst axis j)
      sgn[k]  = +1/-1 with dst_u_j ~= sgn[k]*src_u_k
      M = R_dst^T R_src
    """
    if R_src.shape != (3, 3) or R_dst.shape != (3, 3):
        raise ValueError("R_src and R_dst must be 3x3 with axes as columns.")
    M = R_dst.T @ R_src

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


def map_axis_from_src_to_dst(k_src: int, R_src: np.ndarray, R_dst: np.ndarray) -> Tuple[int, int, Dict[str, Any]]:
    """
    Map an axis index from src to dst.
    Returns:
      k_dst, align_sign, debug
    where dst_u_kdst ~= align_sign * src_u_ksrc
    """
    perm, sgn, M = infer_axis_mapping(R_src, R_dst)
    k_dst = perm[k_src]
    align_sign = sgn[k_src]
    dbg = {"perm": perm, "sign": sgn, "M": M.tolist(), "k_src": int(k_src), "k_dst": int(k_dst), "align_sign": int(align_sign)}
    return int(k_dst), int(align_sign), dbg


def map_face_from_src_to_dst(face_src: str, R_src: np.ndarray, R_dst: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    """
    Map a FACE label from src to dst, including sign flip if axis flips.
    """
    k_src, s_src = parse_face(face_src)
    k_dst, align_sign, dbg = map_axis_from_src_to_dst(k_src, R_src, R_dst)
    s_dst = s_src * align_sign
    face_dst = face_str(k_dst, s_dst)
    dbg.update({"face_src": face_src, "face_dst": face_dst, "s_src": int(s_src), "s_dst": int(s_dst)})
    return face_dst, dbg


# -----------------------------
# OBB ops
# -----------------------------

def translate_obb(pre_obb: Dict[str, Any], t_world: np.ndarray) -> Dict[str, Any]:
    c = np.array(pre_obb["center"], dtype=np.float64)
    c2 = c + np.asarray(t_world, dtype=np.float64)
    return {
        "center": c2.tolist(),
        "axes": pre_obb["axes"],
        "extents": pre_obb["extents"],
    }


def pivot_scale_obb_one_axis(
    pre_obb: Dict[str, Any],
    axis_k: int,
    scale: float,
    pivot_world: np.ndarray,
    min_extent: float = 1e-4,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Scale along one axis (axis_k) by factor `scale`, about pivot_world (anchor),
    so the pivot point stays fixed in world space.

    center shift along u_k:
      t = dot(pivot - c, u_k)
      c' = c + (1 - scale) * t * u_k
    extents:
      e_k' = scale * e_k   (clamped)
    """
    c = np.array(pre_obb["center"], dtype=np.float64)
    R = np.array(pre_obb["axes"], dtype=np.float64)      # columns
    e = np.array(pre_obb["extents"], dtype=np.float64)   # half extents

    if axis_k < 0 or axis_k > 2:
        raise ValueError("axis_k must be 0/1/2")

    u = R[:, axis_k]
    e_old = float(e[axis_k])

    s = float(scale)
    if s <= 0.0:
        # degenerate; clamp
        s = float(min_extent) / max(e_old, 1e-12)

    # clamp extent
    e_new = s * e_old
    if e_new < float(min_extent):
        e_new = float(min_extent)
        s = e_new / max(e_old, 1e-12)

    pivot = np.asarray(pivot_world, dtype=np.float64)
    t = float(np.dot(pivot - c, u))

    c2 = c + (1.0 - s) * t * u

    e2 = e.copy()
    e2[axis_k] = e_new

    post = {
        "center": c2.tolist(),
        "axes": pre_obb["axes"],
        "extents": e2.tolist(),
    }

    info = {
        "axis": int(axis_k),
        "scale_requested": float(scale),
        "scale_applied": float(s),
        "old_extent": float(e_old),
        "new_extent": float(e_new),
        "pivot_world": pivot.tolist(),
        "t_along_axis": float(t),
        "center_shift": (c2 - c).tolist(),
    }
    return post, info


# -----------------------------
# Attachment edge helpers
# -----------------------------

def _faces_for_target(edge: Dict[str, Any], target: str) -> Tuple[str, str, str]:
    """
    Returns:
      (neighbor_name, target_attach_face, neighbor_attach_face)
    """
    a = edge.get("a")
    b = edge.get("b")
    if a == target:
        return b, edge.get("a_face"), edge.get("b_face")
    if b == target:
        return a, edge.get("b_face"), edge.get("a_face")
    raise ValueError("edge does not involve target")


def _case_for_attachment(face_edit: str, face_attach_on_target: str) -> str:
    """
    Case A: same face
    Case B: opposite face
    Case C: orthogonal
    """
    if face_edit == face_attach_on_target:
        return "A"
    if face_edit == opposite_face(face_attach_on_target):
        return "B"
    # orthogonal if axes differ
    kE, _ = parse_face(face_edit)
    kA, _ = parse_face(face_attach_on_target)
    if kE != kA:
        return "C"
    # same axis but not same/opposite shouldn't happen (only +/-) but be safe:
    return "C"


# -----------------------------
# Main API
# -----------------------------

def apply_attachments(
    constraints: Dict[str, Any],
    edit: Dict[str, Any],
    min_extent: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Applies attachment propagation for neighbors of the edited target.

    Returns:
      {
        "<neighbor>": {
          "reason": "attachment",
          "case": "A"|"B"|"C",
          "edge": {...},
          "before_obb": {...},
          "after_obb": {...},
          "debug": {...}
        },
        ...
      }
    """
    nodes = constraints.get("nodes", {}) or {}
    attachments = constraints.get("attachments", []) or []

    target = edit.get("target", None)
    if not target:
        raise ValueError("edit missing 'target'")

    ch = edit.get("change", {}) or {}
    face_edit = ch.get("face", None)
    if face_edit is None:
        raise ValueError("edit['change'] missing 'face'")

    # signed_ratio from target edit (consistent with sym/contain)
    kE, _sE = parse_face(face_edit)
    target_before = nodes[target]["obb"]
    R_T = np.array(target_before["axes"], dtype=np.float64)
    eT = float(target_before["extents"][kE])

    delta_T = ch.get("delta_applied", ch.get("delta_requested", None))
    if delta_T is None:
        raise ValueError("edit['change'] missing delta_applied/delta_requested")

    if eT <= 0:
        raise ValueError(f"target extent on edited axis is non-positive: {eT}")

    signed_ratio = float(delta_T) / float(eT)   # ratio applied along edited axis
    scale_factor = 1.0 + signed_ratio

    anchor_world = ch.get("anchor_world", None)
    # If you didn't store anchor_world in target_face_edit_change.json, we cannot use it.
    # But your edit saver DOES store it (from apply_face_move), so this should exist.
    # Still be robust:
    if anchor_world is None:
        # fallback: if edit has 'after_obb' and 'before_obb', use midpoint of centers
        anchor_world = None

    out: Dict[str, Any] = {}

    if verbose:
        print("\n[AEP][ATTACH] =======================================")
        print(f"[AEP][ATTACH] target={target} face_edit={face_edit} signed_ratio={signed_ratio:+.6f} scale_factor={scale_factor:+.6f}")
        print(f"[AEP][ATTACH] delta_T={float(delta_T):+.6f}  (used for Case A translation)")
        print(f"[AEP][ATTACH] attachments_total={len(attachments)}")
        print("[AEP][ATTACH] =======================================\n")

    # Filter edges involving target
    edges_hit = [e for e in attachments if e.get("a") == target or e.get("b") == target]

    for e in edges_hit:
        neighbor, t_face, n_face = _faces_for_target(e, target)
        if neighbor is None or t_face is None or n_face is None:
            continue
        if neighbor not in nodes or nodes[neighbor].get("obb", None) is None:
            continue

        nb_before = nodes[neighbor]["obb"]
        R_N = np.array(nb_before["axes"], dtype=np.float64)

        case = _case_for_attachment(face_edit, t_face)

        if verbose:
            print(f"[AEP][ATTACH] edge: {target}({t_face}) <-> {neighbor}({n_face})  case={case}")

        # Case B: do nothing
        if case == "B":
            if verbose:
                print("[AEP][ATTACH]  Case B: attachment face is opposite edited face => interface fixed. NO-OP.\n")
            continue

        # Case A: translate neighbor along contact normal by delta_T
        if case == "A":
            # Translate along neighbor's attachment face outward normal (so it stays attached)
            # n_face is +/-u? in neighbor local axes, so normal = sign * u_k
            kN, sN = parse_face(n_face)
            uN = R_N[:, kN]
            t_vec = (sN * float(delta_T)) * uN

            nb_after = translate_obb(nb_before, t_vec)

            if verbose:
                print(f"[AEP][ATTACH]  Case A: translate neighbor by t = {t_vec.tolist()} (delta_T * normal)\n")

            out[neighbor] = {
                "reason": "attachment",
                "case": "A",
                "edge": e,
                "mapped_axis": None,
                "before_obb": nb_before,
                "after_obb": nb_after,
                "debug": {
                    "neighbor_attach_face": n_face,
                    "translation_world": t_vec.tolist(),
                    "delta_T_used": float(delta_T),
                },
            }
            continue

        # Case C: pivot-scale neighbor along mapped axis, preserve anchor_world
        if case == "C":
            # Need anchor_world from attachment edge to preserve contact
            aw = e.get("anchor_world", None)
            if aw is None:
                if verbose:
                    print("[AEP][ATTACH][WARN]  Case C: missing anchor_world on edge. Skip.\n")
                continue
            aw = np.asarray(aw, dtype=np.float64)

            # Map target edited AXIS (kE) onto neighbor axis index
            k_dst, _align_sign, map_dbg = map_axis_from_src_to_dst(kE, R_src=R_T, R_dst=R_N)

            # Apply scaling about anchor
            nb_after, info = pivot_scale_obb_one_axis(
                pre_obb=nb_before,
                axis_k=int(k_dst),
                scale=float(scale_factor),
                pivot_world=aw,
                min_extent=float(min_extent),
            )

            if verbose:
                print(f"[AEP][ATTACH]  Case C: scale neighbor about anchor_world, axis=u{k_dst}, scale={scale_factor:+.6f}")
                print(f"[AEP][ATTACH]         center_shift={info['center_shift']} extent {info['old_extent']:.6f}->{info['new_extent']:.6f}\n")

            out[neighbor] = {
                "reason": "attachment",
                "case": "C",
                "edge": e,
                "mapped_axis": int(k_dst),
                "before_obb": nb_before,
                "after_obb": nb_after,
                "debug": {
                    "target_face_edit": face_edit,
                    "target_attach_face": t_face,
                    "neighbor_attach_face": n_face,
                    "signed_ratio": float(signed_ratio),
                    "scale_factor": float(scale_factor),
                    "anchor_world": aw.tolist(),
                    "axis_mapping": map_dbg,
                    "pivot_scale_info": info,
                },
            }
            continue

    return out
