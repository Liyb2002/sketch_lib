#!/usr/bin/env python3
# AEP/save_json.py
#
# Saves AEP propagation results into a compact JSON schema for visualization.
#
# Fixes:
# - Save BEFORE and AFTER OBB for each changed neighbor:
#     before_obb := constraints["nodes"][name]["obb"]
#     after_obb  := record.after_obb if provided, else computed from before_obb + record
#
# Priority:
#   symmetry > containment > attachment
#
# Attachment neighbor changes are read ONLY from attach_res["changed_nodes"].

import os
import json
from typing import Dict, Any, Optional, Tuple
import copy
import numpy as np


def safe_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ------------------------------------------------------------
# OBB helpers
# ------------------------------------------------------------

def _as_np(x) -> np.ndarray:
    return np.array(x, dtype=np.float64)


def _obb_valid(obb: Any) -> bool:
    return (
        isinstance(obb, dict)
        and isinstance(obb.get("center"), (list, tuple))
        and isinstance(obb.get("axes"), (list, tuple))
        and isinstance(obb.get("extents"), (list, tuple))
        and len(obb.get("center")) == 3
        and len(obb.get("axes")) == 3
        and len(obb.get("extents")) == 3
    )


def _get_node_obb(constraints: Optional[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    if not isinstance(constraints, dict):
        return None
    nodes = constraints.get("nodes", {})
    if not isinstance(nodes, dict):
        return None
    rec = nodes.get(name, {})
    if not isinstance(rec, dict):
        return None
    obb = rec.get("obb")
    return obb if _obb_valid(obb) else None


def _world_to_object(p_world: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    # axes columns are object axes
    return (p_world - origin) @ axes


def _object_to_world(p_local: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    # axes columns are object axes
    return origin + axes @ p_local


def _extract_op_delta_local(op: Any) -> Optional[Tuple[int, float]]:
    """
    Try to parse an operation payload and extract a "scale along axis" delta in LOCAL coords.

    Supported minimal patterns (keep permissive):
      op = {"type":"scale","axis":0,"delta":0.01,...}
      op = {"type":"scale","axis":"u0","delta":0.01,...}
      op = {"op":"scale","axis":0,"delta":0.01,...}
      op = {"kind":"scale", ...}

    Returns:
      (axis_index, delta) where axis_index in {0,1,2}, delta is float,
      or None if cannot parse.
    """
    if not isinstance(op, dict):
        return None

    t = op.get("type", op.get("op", op.get("kind", None)))
    if isinstance(t, str) and t.lower() != "scale":
        return None

    axis = op.get("axis", op.get("u", op.get("dim", None)))
    delta = op.get("delta", op.get("amount", op.get("d", None)))

    if delta is None:
        return None
    try:
        delta_f = float(delta)
    except Exception:
        return None

    axis_idx = None
    if isinstance(axis, int) and axis in (0, 1, 2):
        axis_idx = axis
    elif isinstance(axis, str):
        a = axis.strip().lower()
        if a in ("u0", "x", "0"):
            axis_idx = 0
        elif a in ("u1", "y", "1"):
            axis_idx = 1
        elif a in ("u2", "z", "2"):
            axis_idx = 2

    if axis_idx is None:
        return None

    return axis_idx, delta_f


def _apply_scale_keep_far_face(before_obb: Dict[str, Any], axis_idx: int, signed_delta: float) -> Dict[str, Any]:
    """
    Apply a face-like scaling to an OBB in its local frame:
    - Change the "near" face at +u_axis or -u_axis (depending on signed_delta),
      while keeping the opposite (far) face fixed.
    - This implies center shifts by signed_delta/2 along that axis
      and extent changes by signed_delta/2 (because extents are half-lengths).

    Convention:
      signed_delta > 0 means expanding towards +axis direction (move +face outward)
      signed_delta < 0 means shrinking towards +axis direction (move +face inward)
    This matches "change one face, keep the opposite face unchanged".
    """
    out = copy.deepcopy(before_obb)
    c = _as_np(out["center"])
    R = _as_np(out["axes"])          # 3x3, columns are axes
    e = _as_np(out["extents"])

    # Local displacement vector (in local coords)
    # center shift is delta/2 along axis
    dc_local = np.zeros(3, dtype=np.float64)
    dc_local[axis_idx] = signed_delta / 2.0

    # extent changes by delta/2 (half-lengths)
    e_new = e.copy()
    e_new[axis_idx] = max(1e-9, e_new[axis_idx] + signed_delta / 2.0)

    # transform center shift to world
    dc_world = R @ dc_local
    c_new = c + dc_world

    out["center"] = c_new.tolist()
    out["extents"] = e_new.tolist()
    return out


def _compute_after_obb_from_record(
    before_obb: Optional[Dict[str, Any]],
    reason: str,
    r: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    If r already contains after_obb, return it.
    Otherwise, try to compute after_obb from before_obb and info in r.

    We keep this conservative: only compute when we can parse a scale op.
    For translate ops, attachment.py should already have written after_obb;
    if not, we leave it None so you can detect missing propagation output.
    """
    if not isinstance(r, dict):
        return None

    after = r.get("after_obb", None)
    if _obb_valid(after):
        return after

    if not _obb_valid(before_obb):
        return None

    # Attachment: try to parse scale op and apply
    if reason == "attachment":
        op = r.get("op", None)
        parsed = _extract_op_delta_local(op)
        if parsed is None:
            return None
        axis_idx, delta = parsed
        return _apply_scale_keep_far_face(before_obb, axis_idx, delta)

    # Symmetry / containment:
    # These should normally already provide after_obb.
    # If not, we do NOT guess—return None to expose the bug upstream.
    return None


# ------------------------------------------------------------
# Compacting changes
# ------------------------------------------------------------

def _compact_change(reason: str, r: Dict[str, Any], before_obb: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize different propagation outputs into a compact schema for vis.
    Adds before_obb always; after_obb either from r["after_obb"] or computed.
    """
    if not isinstance(r, dict):
        return {
            "reason": reason,
            "before_obb": before_obb,
            "after_obb": None,
            "debug": {"non_dict_value": str(r)},
        }

    out: Dict[str, Any] = {
        "reason": reason,
        "before_obb": before_obb,
        "after_obb": _compute_after_obb_from_record(before_obb, reason, r),
    }

    # symmetry/containment fields
    if reason in ["symmetry", "containment"]:
        out.update({
            "mapped_face": r.get("mapped_face"),
            "signed_ratio": r.get("signed_ratio"),
            "delta_dst_applied": r.get("delta_dst_applied"),
        })

    # attachment fields
    if reason == "attachment":
        out.update({
            "case": r.get("solving"),
            "kind": r.get("kind"),
            "op": r.get("op"),
        })

        dbg: Dict[str, Any] = {}
        if "edge" in r:
            dbg["edge"] = r.get("edge")
        if "anchor" in r:
            anc = r.get("anchor", {}) if isinstance(r.get("anchor"), dict) else {}
            dbg["anchor"] = {
                "P0": anc.get("P0"),
                "P1": anc.get("P1"),
                "infoA": anc.get("infoA"),
            }
        out["debug"] = dbg if dbg else None

    return out


def save_aep_changes(
    aep_dir: str,
    target_edit: Dict[str, Any],
    symcon_res: Dict[str, Any],
    attach_res: Optional[Dict[str, Any]] = None,
    out_filename: str = "aep_changes.json",
    constraints: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Args:
      aep_dir: sketch/AEP
      target_edit: loaded json from target_face_edit_change.json
      symcon_res: output from apply_symmetry_and_containment(...)
                  {"symmetry": {name: {...}}, "containment": {name: {...}}}
      attach_res: output from apply_attachments(...)
                  {
                    "target": str,
                    "applied": bool,
                    "changed_nodes": { "<neighbor>": {...}, ... },
                    "summary": {...}
                  }
      out_filename: default "aep_changes.json"
      constraints: (optional) full constraints json; if provided, we save
                   before_obb for each changed neighbor from constraints["nodes"][name]["obb"].

    Saves:
      {
        "target": "<label>",
        "target_edit": {...},
        "neighbor_changes": {
          "<neighbor>": {
            "reason": "symmetry"|"containment"|"attachment",
            "before_obb": {...} | null,
            "after_obb": {...} | null,
            ...
          },
          ...
        }
      }
    """
    out_path = os.path.join(aep_dir, out_filename)

    target = target_edit.get("target", None)
    if not target:
        raise ValueError("target_edit missing 'target'")

    neighbor_changes: Dict[str, Any] = {}

    # Priority 1: symmetry
    for name, r in (symcon_res.get("symmetry", {}) or {}).items():
        before = _get_node_obb(constraints, name)
        neighbor_changes[name] = _compact_change("symmetry", r, before)

    # Priority 2: containment
    for name, r in (symcon_res.get("containment", {}) or {}).items():
        if name in neighbor_changes:
            continue
        before = _get_node_obb(constraints, name)
        neighbor_changes[name] = _compact_change("containment", r, before)

    # Priority 3: attachment
    if attach_res is not None:
        changed_nodes = attach_res.get("changed_nodes", {}) if isinstance(attach_res, dict) else {}
        if not isinstance(changed_nodes, dict):
            changed_nodes = {}

        for name, r in changed_nodes.items():
            if name in neighbor_changes:
                continue
            before = _get_node_obb(constraints, name)
            neighbor_changes[name] = _compact_change("attachment", r, before)

    payload = {
        "target": target,
        "target_edit": target_edit,            # full, exact
        "neighbor_changes": neighbor_changes,  # compact per-neighbor changes
    }

    safe_write_json(out_path, payload)
    print("[AEP][SAVE] saved:", out_path)
    print(f"[AEP][SAVE] changed neighbors: {len(neighbor_changes)}")

    # Helpful warning if after_obb is missing
    missing_after = [n for n, rec in neighbor_changes.items()
                     if isinstance(rec, dict) and rec.get("after_obb") is None]
    if missing_after:
        print("[AEP][SAVE][WARN] after_obb missing for:", missing_after)

    return out_path
