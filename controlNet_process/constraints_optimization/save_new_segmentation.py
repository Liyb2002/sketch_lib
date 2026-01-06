#!/usr/bin/env python3
"""
constraints_optimization/save_new_segmentation.py

Save per-label "optimized component" PLYs:
- keep the full heatmap point cloud
- recolor points outside optimized AABB to black
- write <label>.ply into an output folder
"""

import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# -------------------- IO --------------------

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _sanitize_filename(name: str) -> str:
    """
    Safe-ish filename: keep [a-zA-Z0-9_-], convert others to '_'.
    """
    s = str(name).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "unknown"


# -------------------- bbox helpers --------------------

def _extract_box_from_rec(rec: Dict[str, Any], prefer_opt_aabb: bool) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Returns (mn, mx, which) where mn/mx are float64 (3,) arrays.
    which is "opt_aabb_world" or "aabb".
    """
    if prefer_opt_aabb and isinstance(rec.get("opt_aabb_world", None), dict):
        ob = rec["opt_aabb_world"]
        if "min" in ob and "max" in ob:
            mn = np.array(ob["min"], dtype=np.float64).reshape(3)
            mx = np.array(ob["max"], dtype=np.float64).reshape(3)
            return mn, mx, "opt_aabb_world"

    ab = rec.get("aabb", None)
    if isinstance(ab, dict) and "min_bound" in ab and "max_bound" in ab:
        mn = np.array(ab["min_bound"], dtype=np.float64).reshape(3)
        mx = np.array(ab["max_bound"], dtype=np.float64).reshape(3)
        return mn, mx, "aabb"

    return None


def _inside_aabb_mask(pts: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return (
        (pts[:, 0] >= mn[0]) & (pts[:, 0] <= mx[0]) &
        (pts[:, 1] >= mn[1]) & (pts[:, 1] <= mx[1]) &
        (pts[:, 2] >= mn[2]) & (pts[:, 2] <= mx[2])
    )


# -------------------- main API --------------------

def save_optimized_component_plys(
    *,
    pca_bboxes_optimized_json: str,
    out_dir: str,
    prefer_opt_aabb: bool = True,
    overwrite: bool = True,
) -> None:
    """
    For each entry in payload["labels"]:
      - load heatmap PLY (entry["heat_ply"])
      - keep full point cloud
      - recolor OUTSIDE chosen AABB to black
      - save to: out_dir/<label>.ply  (label is sanitized for filename)

    Prints warnings if:
      - heat_ply missing / not found
      - box missing (in that case: saves original pcd without recolor)
    """
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")

    payload = _load_json(pca_bboxes_optimized_json)
    entries = payload.get("labels", []) or []
    if not isinstance(entries, list):
        raise ValueError("Expected payload['labels'] to be a list.")

    os.makedirs(out_dir, exist_ok=True)

    wrote = 0
    missed = 0

    print(f"[SAVE_OPT_PLY] entries: {len(entries)}")
    print(f"[SAVE_OPT_PLY] out_dir: {os.path.abspath(out_dir)}")

    for i, rec in enumerate(entries):
        if not isinstance(rec, dict):
            continue

        label = str(rec.get("label", "unknown"))
        heat_ply = rec.get("heat_ply", None)

        if not isinstance(heat_ply, str) or not heat_ply:
            print(f"[MISS] {label} : missing 'heat_ply' field")
            missed += 1
            continue
        if not os.path.exists(heat_ply):
            print(f"[MISS] {label} : heatmap ply not found at {heat_ply}")
            missed += 1
            continue

        pcd = o3d.io.read_point_cloud(heat_ply)
        if pcd.is_empty():
            print(f"[MISS] {label} : empty point cloud at {heat_ply}")
            missed += 1
            continue

        pts = np.asarray(pcd.points).astype(np.float64)

        # Ensure colors exist; if missing, init to black
        if len(pcd.colors) == len(pcd.points):
            cols = np.asarray(pcd.colors).astype(np.float64)
        else:
            cols = np.zeros((pts.shape[0], 3), dtype=np.float64)

        box = _extract_box_from_rec(rec, prefer_opt_aabb=prefer_opt_aabb)
        if box is None:
            print(f"[WARN] {label} : no usable box; saving original colors (no masking).")
            cols_masked = np.clip(cols, 0.0, 1.0)
            inside_cnt = None
            outside_cnt = None
            box_source = None
        else:
            mn, mx, box_source = box
            inside = _inside_aabb_mask(pts, mn, mx)
            cols_masked = cols.copy()
            cols_masked[~inside] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            cols_masked = np.clip(cols_masked, 0.0, 1.0)
            inside_cnt = int(inside.sum())
            outside_cnt = int((~inside).sum())

        # Build new pcd to avoid mutating original reference unexpectedly
        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(pts)
        out_pcd.colors = o3d.utility.Vector3dVector(cols_masked)

        fname = _sanitize_filename(label) + ".ply"
        out_path = os.path.join(out_dir, fname)

        if (not overwrite) and os.path.exists(out_path):
            print(f"[SKIP] {label} : exists (overwrite=False): {out_path}")
            continue

        ok = o3d.io.write_point_cloud(out_path, out_pcd, write_ascii=True)
        if not ok:
            print(f"[FAIL] {label} : failed to write: {out_path}")
            missed += 1
            continue

        wrote += 1
        if box_source is None:
            print(f"[SAVE] ({i+1}/{len(entries)}) {label} -> {out_path}")
        else:
            print(
                f"[SAVE] ({i+1}/{len(entries)}) {label} -> {out_path}  "
                f"(box={box_source}, inside={inside_cnt}, outside->black={outside_cnt})"
            )

    print(f"\n[SAVE_OPT_PLY] done. wrote={wrote} missed={missed}")
