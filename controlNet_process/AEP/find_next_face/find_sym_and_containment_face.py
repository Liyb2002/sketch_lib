#!/usr/bin/env python3
# AEP/find_next_face/find_sym_and_containment_face.py
#
# SYMMETRY/CONTAINMENT logic: Find corresponding face (same as target's edited face)

from __future__ import annotations
from typing import Any, Dict, Tuple, List
import numpy as np


def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _parse_face(face: str) -> Tuple[int, int]:
    if not isinstance(face, str) or len(face) != 3 or face[0] not in "+-" or face[1] != "u" or face[2] not in "012":
        raise ValueError(f"Invalid face '{face}'. Must be one of: +u0 -u0 +u1 -u1 +u2 -u2")
    s = +1 if face[0] == "+" else -1
    k = int(face[2])
    return k, s


def _face_to_str(axis: int, sign: int) -> str:
    s = "+" if int(sign) >= 0 else "-"
    return f"{s}u{int(axis)}"


def _axes_cols(axes_list: Any) -> np.ndarray:
    """Treat JSON axes as COLUMNS."""
    R = _as_np(axes_list)
    if R.shape != (3, 3):
        raise ValueError(f"axes must be 3x3, got {R.shape}")
    return R


def _unit(v: np.ndarray) -> np.ndarray:
    v = _as_np(v).reshape(3,)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _face_center_world(obb: Dict[str, Any], axis: int, sign: int) -> np.ndarray:
    C = _as_np(obb["center"])
    R = _axes_cols(obb["axes"])
    E = _as_np(obb["extents"])
    return C + float(sign) * float(E[axis]) * R[:, axis]


def _get_target_edit_obbs_and_face(edit: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    target = edit.get("target", None)
    if not isinstance(target, str) or not target:
        raise ValueError("edit missing valid 'target'")

    ch = edit.get("change", {}) or {}
    face = ch.get("face", None)
    if not isinstance(face, str):
        raise ValueError("edit['change'] missing 'face' string")

    before_obb = ch.get("before_obb", None)
    after_obb = ch.get("after_obb", None)
    if not isinstance(before_obb, dict) or not isinstance(after_obb, dict):
        raise ValueError("edit['change'] missing before_obb/after_obb")

    return before_obb, after_obb, face


def _infer_axis_mapping(R_src: np.ndarray, R_dst: np.ndarray) -> Tuple[list, list, np.ndarray]:
    """
    Infer axis mapping between two OBB frames.
    Returns:
      perm[k] = j  (src axis k corresponds to dst axis j)
      sgn[k] = +1/-1  (dst_u_j ~= sgn[k] * src_u_k)
      M = R_dst^T @ R_src (dot matrix)
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


def _map_face_from_src_to_dst(face_src: str, R_src: np.ndarray, R_dst: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    """
    Map face from source OBB to destination OBB using axis alignment.
    Returns: (face_dst, debug_info)
    """
    k_src, s_src = _parse_face(face_src)
    perm, sgn, M = _infer_axis_mapping(R_src, R_dst)

    j_dst = perm[k_src]
    align_sign = sgn[k_src]
    s_dst = s_src * align_sign
    face_dst = _face_to_str(j_dst, s_dst)

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


def find_sym_and_containment_face(
    edit: Dict[str, Any],
    neighbor_before_obb: Dict[str, Any],
    neighbor_after_obb: Dict[str, Any],
    connection_type: str,  # 'symmetry' or 'containment'
    neighbor_name: str = "unknown",  # For debug visualization
) -> List[Dict[str, Any]]:
    """
    Find corresponding face for SYMMETRY/CONTAINMENT connection.
    Uses axis mapping to find the corresponding face in neighbor's frame.
    Returns list of face_edit_change structures (without 'target' field).
    """
    t_before, t_after, t_face = _get_target_edit_obbs_and_face(edit)
    k_t, s_t = _parse_face(t_face)
    
    # Map target's edited face to neighbor's coordinate frame
    R_src = _axes_cols(t_before["axes"])
    R_dst = _axes_cols(neighbor_before_obb["axes"])
    
    neighbor_face, map_debug = _map_face_from_src_to_dst(t_face, R_src, R_dst)
    neighbor_axis = map_debug["j_dst"]
    neighbor_sign = map_debug["s_dst"]
    
    # Get geometry info
    C0 = _as_np(neighbor_before_obb["center"]).reshape(3,)
    C1 = _as_np(neighbor_after_obb["center"]).reshape(3,)
    vC = (C1 - C0)
    vC_norm = float(np.linalg.norm(vC))
    
    R0 = _axes_cols(neighbor_before_obb["axes"])
    R1 = _axes_cols(neighbor_after_obb["axes"])
    E0 = _as_np(neighbor_before_obb["extents"]).reshape(3,)
    E1 = _as_np(neighbor_after_obb["extents"]).reshape(3,)
    
    # Face centers
    p0 = _face_center_world(neighbor_before_obb, neighbor_axis, neighbor_sign)
    p1 = _face_center_world(neighbor_after_obb, neighbor_axis, neighbor_sign)
    
    # Face normal
    n_hat = _unit(R0[:, neighbor_axis] * float(neighbor_sign))
    
    # Delta along normal
    vP = p1 - p0
    delta_face = float(np.dot(vP, n_hat))
    
    # Extent change
    old_extent = float(E0[neighbor_axis])
    new_extent = float(E1[neighbor_axis])
    extent_delta = new_extent - old_extent
    
    # Ratio (based on extent change if significant, else translation)
    if abs(extent_delta) > 1e-8:
        ratio = abs(extent_delta) / max(abs(old_extent), 1e-12)
        expand_or_shrink = "expand" if extent_delta > 0 else "shrink"
    else:
        ratio = abs(delta_face) / max(abs(old_extent), 1e-12)
        expand_or_shrink = "translate"
    
    return [{
        "connection_type": connection_type,
        "change": {
            "type": "move_single_face",
            "delta_ratio_min": float(ratio),
            "delta_ratio_max": float(ratio),
            "ratio_sampled": float(ratio),
            "expand_or_shrink": expand_or_shrink,
            "face": neighbor_face,
            "axis": int(neighbor_axis),
            "axis_name": f"u{int(neighbor_axis)}",
            "sign": int(neighbor_sign),
            "delta_requested": float(delta_face),
            "delta_applied": float(delta_face),
            "old_extent": float(old_extent),
            "new_extent": float(new_extent),
            "min_extent": 1e-4,
            "before_obb": neighbor_before_obb,
            "after_obb": neighbor_after_obb,
            "diagnostics": {
                "selection_reason": f"corresponding_face_{connection_type}",
                "target_edited_face": t_face,
                "neighbor_corresponding_face": neighbor_face,
                "axis_mapping": map_debug,
                "neighbor_center_delta": vC.tolist(),
                "neighbor_center_delta_norm": float(vC_norm),
                "face_center_before": p0.tolist(),
                "face_center_after": p1.tolist(),
                "face_normal_world_before": n_hat.tolist(),
                "extent_delta": float(extent_delta),
            },
            "input_edit_debug": {
                "target_edit_before_obb": t_before,
                "target_edit_after_obb": t_after,
                "target_edit_face": t_face,
            },
        },
    }]