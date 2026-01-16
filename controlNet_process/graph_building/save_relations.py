#!/usr/bin/env python3
# graph_building/save_relations.py

import os
import json
from typing import Dict, Any, List


def _safe_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def save_initial_constraints(
    aep_dir: str,
    symmetry: Dict[str, Any],
    attachments: List[Dict[str, Any]],
    object_space: Dict[str, Any],
    bboxes_by_name: Dict[str, Any],
    params: Dict[str, Any] | None = None,
) -> str:
    """
    Writes:
      <aep_dir>/initial_constraints.json

    Format is intentionally simple + explicit for AEP:
      {
        "params": {...},
        "object_space": {...},
        "symmetry": {
          "groups": {...},
          "pairs":  [...]
        },
        "attachments": [
          {"a","b","distance","anchor_world","anchor_local"},
          ...
        ],
        "nodes": {
          "<label_name>": {"label_id":..., "n_points":..., "obb": {...}},
          ...
        }
      }
    """
    out_path = os.path.join(aep_dir, "initial_constraints.json")

    nodes = {}
    for name, info in bboxes_by_name.items():
        nodes[name] = {
            "label_id": int(info.get("label_id", -1)),
            "n_points": int(info.get("n_points", 0)),
            "obb": info.get("obb_pca", None),
        }

    payload = {
        "params": params or {},
        "object_space": object_space,
        "symmetry": {
            "groups": symmetry.get("groups", {}),
            "pairs": symmetry.get("pairs", []),
        },
        "attachments": attachments,
        "nodes": nodes,
    }

    _safe_write_json(out_path, payload)
    return out_path
