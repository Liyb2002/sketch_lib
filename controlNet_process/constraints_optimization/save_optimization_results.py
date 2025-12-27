#!/usr/bin/env python3
"""
constraints_optimization/save_optimization_results.py

Save per-label optimization results in a clean, inspectable structure.

For each label:
- copy original heatmap PLY
- save original bbox (before)
- save optimized bbox (after)

Output structure:
<OUT_DIR>/optimize_results/
    <label>/
        heat_map_<label>.ply
        bbox_before.json
        bbox_after.json
"""

import os
import json
import shutil
from typing import Dict, Any


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------

def save_optimization_results(
    *,
    heat_dir: str,
    bbox_json_before: str,
    bbox_json_after: str,
    out_dir: str,
) -> None:
    """
    Parameters
    ----------
    heat_dir : str
        Directory containing per-label heat_map_<label>.ply
    bbox_json_before : str
        Original PCA bbox json
    bbox_json_after : str
        Optimized bbox json
    out_dir : str
        Iteration output directory (optimize_iteration/iter_xxx)

    Writes
    ------
    <out_dir>/optimize_results/<label>/*
    """

    payload_before = _load_json(bbox_json_before)
    payload_after = _load_json(bbox_json_after)

    labels_before = payload_before.get("labels", [])
    labels_after = payload_after.get("labels", [])

    if len(labels_before) != len(labels_after):
        raise ValueError(
            "Mismatch between bbox_json_before and bbox_json_after label counts"
        )

    result_root = os.path.join(out_dir, "optimize_results")
    os.makedirs(result_root, exist_ok=True)

    print(f"[SAVE_OPT] Writing optimize_results to: {result_root}")

    for rec_before, rec_after in zip(labels_before, labels_after):
        label = rec_before.get("label", rec_before.get("sanitized", "unknown"))
        label = str(label)

        label_dir = os.path.join(result_root, label)
        os.makedirs(label_dir, exist_ok=True)

        # ---- copy heatmap ----
        heat_ply = rec_before.get("heat_ply", None)
        if heat_ply is None:
            raise ValueError(f"Missing heat_ply for label {label}")

        heat_src = heat_ply
        heat_dst = os.path.join(label_dir, f"heat_map_{label}.ply")

        if not os.path.isfile(heat_src):
            raise FileNotFoundError(f"Heatmap PLY not found: {heat_src}")

        shutil.copyfile(heat_src, heat_dst)

        # ---- save bbox before ----
        bbox_before_out = os.path.join(label_dir, "bbox_before.json")
        _save_json(
            bbox_before_out,
            {
                "label": label,
                "obb": rec_before.get("obb", {}),
                "opt_aabb_world": rec_before.get("opt_aabb_world", None),
                "source": "pca_bboxes",
            },
        )

        # ---- save bbox after ----
        bbox_after_out = os.path.join(label_dir, "bbox_after.json")
        _save_json(
            bbox_after_out,
            {
                "label": label,
                "obb": rec_after.get("obb", {}),
                "opt_aabb_world": rec_after.get("opt_aabb_world", None),
                "opt": rec_after.get("opt", {}),
                "source": "optimizer",
            },
        )

    print(f"[SAVE_OPT] Done. Saved {len(labels_before)} labels.")
