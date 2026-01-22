#!/usr/bin/env python3
# AEP/vis.py
#
# Per-changed-neighbor visualization:
# For each changed neighbor in aep_changes.json, open one window showing:
#   - target OBB: before (blue) + after (red)
#   - neighbor OBB: before (blue) + after (red)
#
# Draw order is ALWAYS:
#   blue first, then red
# so red appears on top when they overlap.
#
# IMPORTANT:
# If object_space is provided (constraints["object_space"]), we assume the OBBs are stored in
# OBJECT SPACE and transform them to WORLD SPACE before drawing on top of overlay PLY.

import os
import json
import copy
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import open3d as o3d


# ----------------------------
# Object-space -> world-space helpers
# ----------------------------

def _get_object_space_T(object_space: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (origin_world, A_obj) where:
      origin_world: (3,)
      A_obj: (3,3) columns are object axes in world coordinates

    object_space json:
      {
        "origin": [..3..],
        "axes": [ [..3..], [..3..], [..3..] ]   # u0,u1,u2 vectors
      }

    We convert list-of-vectors to a matrix with columns = u0,u1,u2.
    """
    origin = np.array(object_space["origin"], dtype=np.float64).reshape(3,)
    axes_list = np.array(object_space["axes"], dtype=np.float64)
    if axes_list.shape != (3, 3):
        raise ValueError("object_space['axes'] must be (3,3)")

    # axes_list is [u0;u1;u2] as rows -> columns matrix
    A_obj = axes_list.T
    return origin, A_obj


def _obb_object_to_world(obb_obj: Dict[str, Any], object_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an OBB defined in object space into world space.

    Assumptions:
      - obb_obj["center"] is in object coords
      - obb_obj["axes"] is a 3x3 basis, with COLUMNS as local axes
      - object_space provides mapping object->world:
          p_world = origin + A_obj @ p_obj
        where A_obj columns are object axes in world.
    """
    origin, A_obj = _get_object_space_T(object_space)

    c_obj = np.array(obb_obj["center"], dtype=np.float64).reshape(3,)

    R_obj = np.array(obb_obj["axes"], dtype=np.float64)
    if R_obj.shape != (3, 3):
        raise ValueError("obb['axes'] must be (3,3)")

    # In your original vis you treat "columns are axes" and pass R directly into open3d.
    R_obj_cols = R_obj

    c_world = origin + (A_obj @ c_obj)
    R_world_cols = A_obj @ R_obj_cols

    return {
        "center": c_world.tolist(),
        "axes": R_world_cols.tolist(),   # still columns are axes
        "extents": obb_obj["extents"],
    }


# ----------------------------
# OBB -> Open3D LineSet
# ----------------------------

def _obb_to_lineset(obb_world: Dict[str, Any]) -> o3d.geometry.LineSet:
    center = np.array(obb_world["center"], dtype=np.float64).reshape(3,)
    R = np.array(obb_world["axes"], dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("obb['axes'] must be (3,3)")

    ext_half = np.array(obb_world["extents"], dtype=np.float64).reshape(3,)
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


def _load_overlay(overlay_ply_path: str) -> Optional[o3d.geometry.PointCloud]:
    if not overlay_ply_path:
        return None
    if not os.path.isfile(overlay_ply_path):
        raise FileNotFoundError(f"Overlay PLY not found: {overlay_ply_path}")
    pcd = o3d.io.read_point_cloud(overlay_ply_path)
    if not pcd.has_points():
        return None
    gray = np.full((len(pcd.points), 3), 0.6, dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(gray)
    return pcd


def _geom_copy(g: o3d.geometry.Geometry) -> o3d.geometry.Geometry:
    # open3d version-safe copy (your build has no .clone())
    return copy.deepcopy(g)


def _maybe_world_obb(obb: Dict[str, Any], object_space: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if object_space is None:
        return obb
    return _obb_object_to_world(obb, object_space)


# ----------------------------
# Public API
# ----------------------------

def vis_from_saved_changes(
    overlay_ply_path: str,
    nodes: Dict[str, Any],
    neighbor_names: List[str],  # kept for API compatibility; ignored
    aep_changes_json: str,
    target: str,
    window_name: str = "AEP: per-changed-neighbor view",
    show_overlay: bool = True,
    object_space: Optional[Dict[str, Any]] = None,  # pass constraints["object_space"]
) -> None:
    """
    For each changed neighbor in aep_changes.json:
      - overlay (gray) optional
      - target before (blue) + target after (red)
      - neighbor before (blue) + neighbor after (red)

    Draw order ALWAYS:
      blue first, then red
    so red appears on top.

    If object_space is provided:
      treat ALL obbs as OBJECT-SPACE and transform to WORLD-SPACE for drawing.
    """
    if not os.path.isfile(aep_changes_json):
        raise FileNotFoundError(f"Missing aep_changes.json: {aep_changes_json}")

    changes = _load_json(aep_changes_json)
    neighbor_changes = changes.get("neighbor_changes", {}) or {}

    target_edit = changes.get("target_edit", {}) or {}
    target_change = (target_edit.get("change", {}) or {})
    target_after_raw = target_change.get("after_obb", None)

    target_before_raw = None
    if target in nodes and isinstance(nodes[target], dict):
        target_before_raw = nodes[target].get("obb", None)

    if target_before_raw is None:
        print(f"[AEP][VIS][WARN] target '{target}' missing in nodes or has no obb. Target will be skipped in all views.")
    if target_after_raw is None:
        print("[AEP][VIS][WARN] target_edit.change.after_obb missing in aep_changes.json. Target-after will be skipped in all views.")

    overlay = _load_overlay(overlay_ply_path) if show_overlay else None

    changed_names = list(neighbor_changes.keys())
    if len(changed_names) == 0:
        print("[AEP][VIS] No changed neighbors found in aep_changes.json -> nothing to visualize.")
        return

    # Pre-transform target boxes once (reuse)
    target_before = _maybe_world_obb(target_before_raw, object_space) if target_before_raw is not None else None
    target_after = _maybe_world_obb(target_after_raw, object_space) if target_after_raw is not None else None

    for i, nb in enumerate(changed_names):
        nb_change = neighbor_changes.get(nb, {}) or {}
        nb_after_raw = nb_change.get("after_obb", None)

        nb_before_raw = None
        if nb in nodes and isinstance(nodes[nb], dict):
            nb_before_raw = nodes[nb].get("obb", None)

        if nb_before_raw is None:
            print(f"[AEP][VIS][WARN] neighbor '{nb}' missing in nodes or has no obb. (skip this neighbor)")
            continue
        if nb_after_raw is None:
            print(f"[AEP][VIS][WARN] neighbor_changes['{nb}'].after_obb missing. (skip this neighbor)")
            continue

        nb_before = _maybe_world_obb(nb_before_raw, object_space)
        nb_after = _maybe_world_obb(nb_after_raw, object_space)

        geoms: List[o3d.geometry.Geometry] = []
        if overlay is not None:
            geoms.append(_geom_copy(overlay))

        # ---------- TARGET (blue then red) ----------
        if target_before is not None:
            ls_tb = _paint_lineset(_obb_to_lineset(target_before), (0.0, 0.0, 1.0))  # blue
            geoms.append(ls_tb)
        if target_after is not None:
            ls_ta = _paint_lineset(_obb_to_lineset(target_after), (1.0, 0.0, 0.0))    # red
            geoms.append(ls_ta)

        # ---------- NEIGHBOR (blue then red) ----------
        ls_nb = _paint_lineset(_obb_to_lineset(nb_before), (0.0, 0.0, 1.0))          # blue
        geoms.append(ls_nb)

        ls_na = _paint_lineset(_obb_to_lineset(nb_after), (1.0, 0.0, 0.0))           # red
        geoms.append(ls_na)

        reason = nb_change.get("reason", "unknown")
        case = nb_change.get("case", None)
        tag = f"{reason}" + (f" | case={case}" if case is not None else "")

        wn = f"{window_name} [{i+1}/{len(changed_names)}] target={target} neighbor={nb} | {tag}"
        o3d.visualization.draw_geometries(geoms, window_name=wn)
