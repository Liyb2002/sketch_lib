#!/usr/bin/env python3
# constraints_optimization/vis_preopt_label_boxes_heatmap.py
"""
vis_preopt_label_boxes_heatmap.py

Pre-optimization visualization (AABB, optimizer-style):

For EACH label:
  - load saved heatmap PLY:
      heat_dir/heatmaps/<label>/heat_map_<label>.ply
    This is the whole shape colored by heat fraction for that label.
  - overlay bounding boxes read EXACTLY like no_overlapping.py:
      c = parameters.center
      e = abs(parameters.extent)
      mn = c - 0.5*e
      mx = c + 0.5*e
    If parameters.aabb_min/aabb_max exist, prefer them.
"""

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# ---------------- IO + helpers ----------------

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _sanitize_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_\-]+", "", name)
    return name or "unknown"

def _heat_ply_path(heat_dir: str, label: str) -> str:
    s = _sanitize_name(label)
    return os.path.join(heat_dir, "heatmaps", s, f"heat_map_{s}.ply")

def _load_primitives(primitives_json_path: str) -> List[Dict[str, Any]]:
    raw = _load_json(primitives_json_path)
    if isinstance(raw, dict) and "primitives" in raw:
        return raw["primitives"]
    if isinstance(raw, list):
        return raw
    raise ValueError(f"Unexpected primitives JSON format: {primitives_json_path}")


# ---------------- box reading (LEARNED FROM no_overlapping.py) ----------------

def _get_aabb_from_primitive_optimizer_style(p: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    EXACTLY matches how your no_overlapping.py reads boxes:
      c = center
      e = abs(extent)
      mn = c - 0.5*e
      mx = c + 0.5*e
      ordered with min/max

    If aabb_min/aabb_max exist, prefer them (already computed somewhere else).
    """
    params = p.get("parameters", {}) or {}

    if "aabb_min" in params and "aabb_max" in params:
        mn = np.asarray(params["aabb_min"], dtype=np.float64).reshape(3)
        mx = np.asarray(params["aabb_max"], dtype=np.float64).reshape(3)
        return np.minimum(mn, mx), np.maximum(mn, mx)

    c = params.get("center", None)
    e = params.get("extent", None)
    if c is None or e is None:
        return None

    c = np.asarray(c, dtype=np.float64).reshape(3)
    e = np.abs(np.asarray(e, dtype=np.float64).reshape(3))

    if not np.isfinite(c).all() or not np.isfinite(e).all():
        return None

    mn = c - 0.5 * e
    mx = c + 0.5 * e

    mn2 = np.minimum(mn, mx)
    mx2 = np.maximum(mn, mx)
    return mn2, mx2


# ---------------- Open3D drawing ----------------

def _aabb_lineset(mn: np.ndarray, mx: np.ndarray, color_rgb=(0.2, 0.6, 1.0)) -> "o3d.geometry.LineSet":
    mn = np.asarray(mn, dtype=np.float64).reshape(3)
    mx = np.asarray(mx, dtype=np.float64).reshape(3)

    corners = np.array([
        [mn[0], mn[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]],
        [mn[0], mx[1], mx[2]],
    ], dtype=np.float64)

    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)

    col = np.tile(np.array(color_rgb, dtype=np.float64)[None, :], (edges.shape[0], 1))
    ls.colors = o3d.utility.Vector3dVector(col)
    return ls


# ---------------- Public API ----------------

def vis_preopt_label_boxes_heatmap(
    *,
    primitives_json_path: str,
    heat_dir: str,
    max_labels: Optional[int] = None,
    max_boxes_per_label: Optional[int] = None,
) -> None:
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")

    if not os.path.exists(primitives_json_path):
        raise FileNotFoundError(f"[VIS][FATAL] Missing primitives json: {primitives_json_path}")

    summary_json = os.path.join(heat_dir, "heatmaps_summary.json")
    if not os.path.exists(summary_json):
        raise FileNotFoundError(
            f"[VIS][FATAL] Missing heatmaps_summary.json:\n  {summary_json}\n"
            "Heatmaps must already exist in this directory."
        )

    prims = _load_primitives(primitives_json_path)

    # group by label
    label_to_prims: Dict[str, List[Dict[str, Any]]] = {}
    for p in prims:
        lab = str(p.get("label", "unknown"))
        label_to_prims.setdefault(lab, []).append(p)

    labels = sorted(label_to_prims.keys())
    if max_labels is not None:
        labels = labels[: int(max_labels)]

    print(f"[VIS] labels in primitives: {len(labels)}")
    print(f"[VIS] heat_dir: {heat_dir}")

    for lab in labels:
        heat_ply = _heat_ply_path(heat_dir, lab)
        if not os.path.exists(heat_ply):
            print(f"[VIS][WARN] Missing heat PLY for label='{lab}': {heat_ply}")
            continue

        # Heatmap-colored whole shape for this label
        pcd = o3d.io.read_point_cloud(heat_ply)
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            print(f"[VIS][WARN] Empty heatmap PLY for label='{lab}': {heat_ply}")
            continue

        geoms = [pcd]

        prim_list = label_to_prims.get(lab, [])
        if max_boxes_per_label is not None:
            prim_list = prim_list[: int(max_boxes_per_label)]

        boxes = 0
        skipped = 0
        for p in prim_list:
            aabb = _get_aabb_from_primitive_optimizer_style(p)
            if aabb is None:
                skipped += 1
                continue
            mn, mx = aabb
            geoms.append(_aabb_lineset(mn, mx, color_rgb=(0.2, 0.6, 1.0)))  # blue
            boxes += 1

        title = f"PRE-OPT | label={lab} | AABB boxes={boxes} (skipped={skipped}) | heatmap overlay"
        print(f"[VIS] opening: {title}")
        o3d.visualization.draw_geometries(geoms, window_name=title)

    print("[VIS] done.")
