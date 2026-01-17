#!/usr/bin/env python3
# AEP/save_json.py
#
# Save AEP changes (target edit + propagated neighbor edits) into ONE json file
# under sketch/AEP/, so vis can read it later.
#
# Output file:
#   sketch/AEP/aep_changes.json
#
# Contents:
# - target_edit (exact, from target_face_edit_change.json)
# - neighbor_changes: merged from
#     - symmetry propagation
#     - containment propagation
#     - attachment propagation
#
# Priority (if same neighbor changed by multiple):
#   symmetry > containment > attachment
# (you can change the order below if you want)

import os
import json
from typing import Dict, Any, Optional


def safe_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _compact_change(reason: str, r: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize different propagation outputs into a compact schema for vis.
    """
    out = {
        "reason": reason,
        "after_obb": r.get("after_obb"),
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
            "case": r.get("case"),
            "mapped_axis": r.get("mapped_axis"),
            # optional debug fields if present
            "debug": r.get("debug", None),
        })

    return out


def save_aep_changes(
    aep_dir: str,
    target_edit: Dict[str, Any],
    symcon_res: Dict[str, Any],
    attach_res: Optional[Dict[str, Any]] = None,
    out_filename: str = "aep_changes.json",
) -> str:
    """
    Args:
      aep_dir: sketch/AEP
      target_edit: loaded json from target_face_edit_change.json
      symcon_res: output from apply_symmetry_and_containment(...)
                  {"symmetry": {name: {...}}, "containment": {name: {...}}}
      attach_res: output from apply_attachments(...)
                  {name: {...}, ...}
      out_filename: default "aep_changes.json"

    Saves:
      {
        "target": "<label>",
        "target_edit": {...},
        "neighbor_changes": {
          "<neighbor>": {
            "reason": "symmetry"|"containment"|"attachment",
            "after_obb": {...},
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
        neighbor_changes[name] = _compact_change("symmetry", r)

    # Priority 2: containment (skip if symmetry already changed it)
    for name, r in (symcon_res.get("containment", {}) or {}).items():
        if name in neighbor_changes:
            continue
        neighbor_changes[name] = _compact_change("containment", r)

    # Priority 3: attachment (skip if already changed by sym/contain)
    if attach_res is not None:
        for name, r in (attach_res or {}).items():
            if name in neighbor_changes:
                continue
            neighbor_changes[name] = _compact_change("attachment", r)

    payload = {
        "target": target,
        "target_edit": target_edit,          # full, exact
        "neighbor_changes": neighbor_changes # changed neighbors (red boxes)
    }

    safe_write_json(out_path, payload)
    print("[AEP][SAVE] saved:", out_path)
    print(f"[AEP][SAVE] changed neighbors: {len(neighbor_changes)}")
    return out_path
