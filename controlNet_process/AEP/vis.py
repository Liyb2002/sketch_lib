#!/usr/bin/env python3
# AEP/vis.py

import os
import json
from typing import Dict, Any, List

import numpy as np
import open3d as o3d


def _obb_to_lineset(obb: Dict[str, Any]) -> o3d.geometry.LineSet:
    center = np.array(obb["center"], dtype=np.float64)
    R = np.array(obb["axes"], dtype=np.float64)          # columns are axes
    ext_half = np.array(obb["extents"], dtype=np.float64)
    ext_full = 2.0 * ext_half
    o3d_obb = o3d.geometry.OrientedBoundingBox(center, R, ext_full)
    return o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_obb)


def _paint_lineset(ls: o3d.geometry.LineSet, rgb) -> o3d.geometry.LineSet:
    colors = np.tile(np.array(rgb, dtype=np.float64)[None, :], (len(ls.lines), 1))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def vis_from_saved_changes(
    overlay_ply_path: str,
    nodes: Dict[str, Any],
    neighbor_names: List[str],
    aep_changes_json: str,
    target: str,
    window_name: str = "AEP: target+neighbors (blue) + changed (red)",
    show_overlay: bool = True,
) -> None:
    """
    Draw:
      - overlay (gray)
      - target before (blue) + target after (red)
      - neighbors before (blue)
      - changed neighbors after (red), based on aep_changes.json
    """
    if not os.path.isfile(aep_changes_json):
        raise FileNotFoundError(f"Missing aep_changes.json: {aep_changes_json}")

    changes = _load_json(aep_changes_json)
    neighbor_changes = changes.get("neighbor_changes", {}) or {}
    target_edit = changes.get("target_edit", {}) or {}
    target_change = (target_edit.get("change", {}) or {})
    target_after = target_change.get("after_obb", None)

    geoms: List[o3d.geometry.Geometry] = []

    # overlay
    if show_overlay:
        if not os.path.isfile(overlay_ply_path):
            raise FileNotFoundError(f"Overlay PLY not found: {overlay_ply_path}")
        pcd = o3d.io.read_point_cloud(overlay_ply_path)
        if pcd.has_points():
            gray = np.full((len(pcd.points), 3), 0.6, dtype=np.float64)
            pcd.colors = o3d.utility.Vector3dVector(gray)
        geoms.append(pcd)

    # ---------------------------
    # TARGET: blue before + red after
    # ---------------------------
    if target not in nodes or nodes[target].get("obb", None) is None:
        print(f"[AEP][VIS][WARN] target '{target}' missing in nodes or has no obb. (skip target vis)")
    else:
        target_before = nodes[target]["obb"]
        geoms.append(_paint_lineset(_obb_to_lineset(target_before), (0.0, 0.0, 1.0)))  # blue

        if target_after is not None:
            geoms.append(_paint_lineset(_obb_to_lineset(target_after), (1.0, 0.0, 0.0)))  # red
        else:
            print("[AEP][VIS][WARN] target_edit.change.after_obb missing in aep_changes.json (skip target red)")

    # ---------------------------
    # NEIGHBORS: blue before + red after (if changed)
    # ---------------------------
    for name in neighbor_names:
        if name == target:
            continue  # already drawn above

        info = nodes.get(name, None)
        if info is None:
            continue
        obb_before = info.get("obb", None)
        if obb_before is None:
            continue

        geoms.append(_paint_lineset(_obb_to_lineset(obb_before), (0.0, 0.0, 1.0)))  # blue

        if name in neighbor_changes:
            after_obb = neighbor_changes[name].get("after_obb", None)
            if after_obb is not None:
                geoms.append(_paint_lineset(_obb_to_lineset(after_obb), (1.0, 0.0, 0.0)))  # red

    o3d.visualization.draw_geometries(geoms, window_name=window_name)
