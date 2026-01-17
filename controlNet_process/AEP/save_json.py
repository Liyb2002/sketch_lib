#!/usr/bin/env python3
# AEP/save_json.py
#
# Save AEP changes (target edit + propagated neighbor edits) into ONE json file
# under sketch/AEP/, so vis can read it later.
#
# Output file:
#   sketch/AEP/aep_changes.json
#
# This file intentionally stores:
# - target_edit (exact, from target_face_edit_change.json)
# - neighbor_changes (only the neighbors that got changed by sym/contain for now)
#
# NOTE: we store ONLY the "after_obb" for changed nodes (and optionally their before_obb for debugging).

import os
import json
from typing import Dict, Any


def safe_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def save_aep_changes(
    aep_dir: str,
    target_edit: Dict[str, Any],
    symcon_res: Dict[str, Any],
    out_filename: str = "aep_changes.json",
) -> str:
    """
    Args:
      aep_dir: sketch/AEP
      target_edit: loaded json from target_face_edit_change.json
      symcon_res: output from apply_symmetry_and_containment(...)
                 format: {"symmetry": {name: {...}}, "containment": {name: {...}}}

    Saves:
      aep_changes.json with:
        {
          "target": "<label>",
          "target_edit": {... full content of target_face_edit_change.json ...},
          "neighbor_changes": {
             "<neighbor>": {
                "reason": "symmetry" | "containment",
                "mapped_face": "...",
                "signed_ratio": ...,
                "delta_dst_applied": ...,
                "after_obb": {...}
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

    for name, r in (symcon_res.get("symmetry", {}) or {}).items():
        neighbor_changes[name] = {
            "reason": "symmetry",
            "mapped_face": r.get("mapped_face"),
            "signed_ratio": r.get("signed_ratio"),
            "delta_dst_applied": r.get("delta_dst_applied"),
            "after_obb": r.get("after_obb"),
        }

    for name, r in (symcon_res.get("containment", {}) or {}).items():
        # if already set by symmetry, keep symmetry (it has higher priority for now)
        if name in neighbor_changes:
            continue
        neighbor_changes[name] = {
            "reason": "containment",
            "mapped_face": r.get("mapped_face"),
            "signed_ratio": r.get("signed_ratio"),
            "delta_dst_applied": r.get("delta_dst_applied"),
            "after_obb": r.get("after_obb"),
        }

    payload = {
        "target": target,
        "target_edit": target_edit,          # full, exact
        "neighbor_changes": neighbor_changes # only changed neighbors (red boxes)
    }

    safe_write_json(out_path, payload)
    print("[AEP][SAVE] saved:", out_path)
    print(f"[AEP][SAVE] changed neighbors: {len(neighbor_changes)}")
    return out_path
