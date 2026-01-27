#!/usr/bin/env python3
# AEP/find_next_face/find_attachment_face.py
#
# ATTACHMENT logic: Find the passive face (opposite of attachment face)

from __future__ import annotations
from typing import Any, Dict, Tuple, List
import numpy as np
import copy


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


def _all_face_centers_world(obb: Dict[str, Any]) -> List[Tuple[int, int, np.ndarray]]:
    out = []
    for axis in (0, 1, 2):
        for sign in (-1, +1):
            out.append((axis, sign, _face_center_world(obb, axis, sign)))
    return out


def _axes_close(R0: np.ndarray, R1: np.ndarray, tol: float) -> Tuple[bool, float]:
    d = float(np.linalg.norm(_as_np(R0) - _as_np(R1)))
    return (d <= tol), d


def _extents_close(E0: np.ndarray, E1: np.ndarray, tol: float) -> Tuple[bool, np.ndarray]:
    E0 = _as_np(E0).reshape(3,)
    E1 = _as_np(E1).reshape(3,)
    d = E1 - E0
    ok = bool(np.all(np.abs(d) <= tol))
    return ok, d


def _translated_obb_along(obb_before: Dict[str, Any], delta: float, n_hat: np.ndarray) -> Dict[str, Any]:
    out = copy.deepcopy(obb_before)
    C0 = _as_np(obb_before["center"]).reshape(3,)
    out["center"] = (C0 + float(delta) * _unit(n_hat)).tolist()
    return out


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


def find_attachment_face(
    edit: Dict[str, Any],
    neighbor_before_obb: Dict[str, Any],
    neighbor_after_obb: Dict[str, Any],
    *,
    extents_tol: float = 1e-6,
    axes_tol: float = 1e-6,
    min_translation: float = 1e-8,
    normal_alignment_min: float = 0.95,
) -> Dict[str, Any]:
    """
    Find passive face for ATTACHMENT connection.
    Returns face_edit_change structure (without 'target' field).
    """
    t_before, t_after, t_face = _get_target_edit_obbs_and_face(edit)
    k_t, s_t = _parse_face(t_face)

    t_face_center0 = _face_center_world(t_before, axis=k_t, sign=s_t)

    # Find attachment face = nearest neighbor face center (BEFORE)
    best = None
    best_dist = 1e30
    for axis, sign, p in _all_face_centers_world(neighbor_before_obb):
        d = float(np.linalg.norm(p - t_face_center0))
        if d < best_dist:
            best_dist = d
            best = (axis, sign, p)
    if best is None:
        raise RuntimeError("Failed to infer attachment face")

    attached_axis, attached_sign, _ = best
    attached_face = _face_to_str(int(attached_axis), int(attached_sign))

    # Raw neighbor motion
    C0 = _as_np(neighbor_before_obb["center"]).reshape(3,)
    C1_raw = _as_np(neighbor_after_obb["center"]).reshape(3,)
    vC = (C1_raw - C0)
    vC_norm = float(np.linalg.norm(vC))

    # Translation-only checks (raw)
    R0 = _axes_cols(neighbor_before_obb["axes"])
    R1_raw = _axes_cols(neighbor_after_obb["axes"])
    E0 = _as_np(neighbor_before_obb["extents"]).reshape(3,)
    E1_raw = _as_np(neighbor_after_obb["extents"]).reshape(3,)

    ext_ok, ext_delta = _extents_close(E0, E1_raw, tol=extents_tol)
    axes_ok, axes_diff = _axes_close(R0, R1_raw, tol=axes_tol)

    # Candidates (exclude attachment face)
    candidates = []
    for axis in (0, 1, 2):
        for sign in (-1, +1):
            if int(axis) == int(attached_axis) and int(sign) == int(attached_sign):
                continue
            n_hat = _unit(R0[:, int(axis)] * float(sign))
            delta = float(np.dot(vC, n_hat))
            align = float(abs(delta) / max(vC_norm, 1e-12)) if vC_norm > 1e-12 else 0.0
            tang = float(np.linalg.norm(vC - delta * n_hat))
            candidates.append({
                "axis": int(axis),
                "sign": int(sign),
                "face": _face_to_str(int(axis), int(sign)),
                "n_hat": n_hat,
                "delta": delta,
                "alignment": align,
                "tangential": tang,
            })

    valid = []
    for c in candidates:
        if vC_norm < min_translation:
            continue
        if c["alignment"] < float(normal_alignment_min):
            continue
        if not ext_ok or not axes_ok:
            continue
        valid.append(c)

    if valid:
        valid.sort(key=lambda x: abs(float(x["delta"])), reverse=True)
        chosen = valid[0]
        chosen_reason = "translation_only_valid"
    else:
        # Fallback: opposite-of-attachment
        passive_axis = int(attached_axis)
        passive_sign = int(-attached_sign)
        n_hat = _unit(R0[:, passive_axis] * float(passive_sign))
        delta = float(np.dot(vC, n_hat))
        chosen = {
            "axis": passive_axis,
            "sign": passive_sign,
            "face": _face_to_str(passive_axis, passive_sign),
            "n_hat": n_hat,
            "delta": delta,
            "alignment": float(abs(delta) / max(vC_norm, 1e-12)) if vC_norm > 1e-12 else 0.0,
            "tangential": float(np.linalg.norm(vC - delta * n_hat)),
        }
        chosen_reason = "fallback_opposite_of_attachment"

    axis = int(chosen["axis"])
    sign = int(chosen["sign"])
    face = str(chosen["face"])
    n_hat = _as_np(chosen["n_hat"]).reshape(3,)
    delta_face = float(chosen["delta"])

    # Fixed translation-only after
    neighbor_after_obb_fixed = _translated_obb_along(neighbor_before_obb, delta=delta_face, n_hat=n_hat)

    # Diagnostics
    p0 = _face_center_world(neighbor_before_obb, axis, sign)
    p1_fixed = _face_center_world(neighbor_after_obb_fixed, axis, sign)

    old_extent = float(E0[axis])
    ratio = abs(delta_face) / max(abs(old_extent), 1e-12)

    return {
        "connection_type": "attachment",
        "change": {
            "type": "move_single_face",
            "delta_ratio_min": float(ratio),
            "delta_ratio_max": float(ratio),
            "ratio_sampled": float(ratio),
            "expand_or_shrink": "translate",
            "face": face,
            "axis": int(axis),
            "axis_name": f"u{int(axis)}",
            "sign": int(sign),
            "delta_requested": float(delta_face),
            "delta_applied": float(delta_face),
            "old_extent": float(old_extent),
            "new_extent": float(old_extent),
            "min_extent": 1e-4,
            "before_obb": neighbor_before_obb,
            "after_obb": neighbor_after_obb_fixed,
            "diagnostics": {
                "selection_reason": str(chosen_reason),
                "attached_face": attached_face,
                "attached_face_dist_to_target_edited_face_center": float(best_dist),
                "target_edited_face": t_face,
                "target_edited_face_center_before": t_face_center0.tolist(),
                "chosen_face_alignment": float(chosen.get("alignment", 0.0)),
                "chosen_face_tangential_component": float(chosen.get("tangential", 0.0)),
                "neighbor_center_delta_raw": vC.tolist(),
                "neighbor_center_delta_norm_raw": float(vC_norm),
                "passive_face_center_before": p0.tolist(),
                "passive_face_center_after_fixed": p1_fixed.tolist(),
                "passive_face_normal_world_before": _unit(n_hat).tolist(),
                "raw_geometry_checks": {
                    "extents_tol": float(extents_tol),
                    "axes_tol": float(axes_tol),
                    "extents_unchanged_raw": bool(ext_ok),
                    "extents_delta_raw": ext_delta.tolist(),
                    "axes_unchanged_raw": bool(axes_ok),
                    "axes_diff_fro_raw": float(axes_diff),
                },
                "neighbor_after_obb_raw": neighbor_after_obb,
            },
            "input_edit_debug": {
                "target_edit_before_obb": t_before,
                "target_edit_after_obb": t_after,
                "target_edit_face": t_face,
            },
        },
    }