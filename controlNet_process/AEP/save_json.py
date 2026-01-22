#!/usr/bin/env python3
# AEP/save_json.py
#
# Fixes your crash:
#   AttributeError: 'str' object has no attribute 'get'
#
# Why it happened:
# - attach_res is NOT {name: {...}}.
# - attach_res is the full dict returned by apply_attachments():
#     {
#       "target": ...,
#       "applied": ...,
#       "changed_nodes": { "<neighbor>": {...rec...}, ... },
#       "summary": {...}
#     }
# - Your old save code iterated: for name, r in (attach_res or {}).items()
#   which yields keys like "target", "applied", "changed_nodes", "summary".
#   Some of those values are strings/bools, so r.get(...) crashes.
#
# Updated behavior:
# - Only read attachment neighbor changes from attach_res["changed_nodes"].
# - Keep priority: symmetry > containment > attachment.
# - Update _compact_change() for attachment records produced by your new attachment.py:
#     r contains: kind, solving, after_obb, op, edge, (optional) anchor
#   We store:
#     case := r["solving"]
#     kind := r["kind"]
#     op   := r["op"]
#     debug := small debug bundle (edge/anchor optional)

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

    Expected inputs:
      - symmetry/containment item r: dict with after_obb + mapping fields
      - attachment item r: dict with after_obb + kind/solving/op + optional debug
    """
    if not isinstance(r, dict):
        # be defensive (never crash save)
        return {"reason": reason, "after_obb": None, "debug": {"non_dict_value": str(r)}}

    out: Dict[str, Any] = {
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

    # attachment fields (new attachment.py format)
    if reason == "attachment":
        out.update({
            "case": r.get("solving"),      # e.g. A1, A3, B1->A1, P1+P2(rigid)
            "kind": r.get("kind"),         # face/volume/point
            "op": r.get("op"),             # translate/scale + params
        })

        # optional compact debug
        dbg: Dict[str, Any] = {}
        if "edge" in r:
            dbg["edge"] = r.get("edge")
        if "anchor" in r:
            # anchor can be big; keep only key bits
            anc = r.get("anchor", {}) if isinstance(r.get("anchor"), dict) else {}
            dbg["anchor"] = {
                "P0": anc.get("P0"),
                "P1": anc.get("P1"),
                "infoA": anc.get("infoA"),
            }
        if dbg:
            out["debug"] = dbg
        else:
            out["debug"] = None

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
                  {
                    "target": str,
                    "applied": bool,
                    "changed_nodes": { "<neighbor>": {...}, ... },
                    "summary": {...}
                  }
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
        # IMPORTANT FIX: only iterate over attach_res["changed_nodes"]
        changed_nodes = attach_res.get("changed_nodes", {}) if isinstance(attach_res, dict) else {}
        if not isinstance(changed_nodes, dict):
            changed_nodes = {}

        for name, r in changed_nodes.items():
            if name in neighbor_changes:
                continue
            neighbor_changes[name] = _compact_change("attachment", r)

    payload = {
        "target": target,
        "target_edit": target_edit,           # full, exact
        "neighbor_changes": neighbor_changes, # changed neighbors (red boxes)
    }

    safe_write_json(out_path, payload)
    print("[AEP][SAVE] saved:", out_path)
    print(f"[AEP][SAVE] changed neighbors: {len(neighbor_changes)}")
    return out_path
