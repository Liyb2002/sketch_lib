#!/usr/bin/env python3
# vis_changed_with_overlay_autoframe.py
#
# Standalone visualization (hard-coded paths) that AUTO-DETECTS whether
# OBBs in aep_changes.json are already WORLD-SPACE or need OBJECT->WORLD transform.
#
# It uses the fact that aep_changes.json contains BOTH:
#   - op.debug.volume_info.anchor_local   (object space)
#   - debug.edge.anchor_world             (world space)
#
# We test whether:
#   origin + A_obj @ anchor_local  ~= anchor_world
# If yes -> object_space is OBJECT->WORLD and OBBs are likely OBJECT space.
# Otherwise -> assume OBBs are already WORLD space (no transform).
#
# Then for EACH changed neighbor in aep_changes.json:
#   - Draw overlay (gray)
#   - Draw target before (blue) + after (red)
#   - Draw neighbor before (blue) + after (red)
# Always BLUE first then RED.

import os
import json
import copy
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import open3d as o3d


# ----------------------------
# Hard-coded paths
# ----------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))

AEP_DATA_DIR = os.path.join(ROOT, "sketch", "AEP")
CONSTRAINTS_PATH = os.path.join(AEP_DATA_DIR, "filtered_relations.json")
AEP_CHANGES_PATH = os.path.join(AEP_DATA_DIR, "aep_changes.json")

OVERLAY_PLY = os.path.join(
    ROOT, "sketch", "partfield_overlay", "label_assignment_k20", "assignment_colored.ply"
)


# ----------------------------
# IO helpers
# ----------------------------

def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return json.load(f)


def _geom_copy(g: o3d.geometry.Geometry) -> o3d.geometry.Geometry:
    return copy.deepcopy(g)


def _load_overlay_ply_gray(path: str) -> Optional[o3d.geometry.PointCloud]:
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Overlay PLY not found: {path}")
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        return None
    gray = np.full((len(pcd.points), 3), 0.6, dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(gray)
    return pcd


# ----------------------------
# Object-space transform helpers
# ----------------------------

def _get_object_space_T(object_space: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (origin_world, A_obj) where columns of A_obj are object axes in world coords.

    object_space:
      {
        "origin": [x,y,z],
        "axes": [ [u0x,u0y,u0z], [u1x,...], [u2x,...] ]   # u0,u1,u2 as rows
      }
    """
    origin = np.array(object_space["origin"], dtype=np.float64).reshape(3,)
    axes_list = np.array(object_space["axes"], dtype=np.float64)
    if axes_list.shape != (3, 3):
        raise ValueError("constraints['object_space']['axes'] must be (3,3)")
    A_obj = axes_list.T  # rows -> columns
    return origin, A_obj


def _obb_object_to_world(obb_obj: Dict[str, Any], origin: np.ndarray, A_obj: np.ndarray) -> Dict[str, Any]:
    """
    OBB in object space -> world space.

    Assumptions:
      - obb["center"] is object coords
      - obb["axes"] is 3x3 with COLUMNS as local axes (in object coords)
      - p_world = origin + A_obj @ p_obj
      - R_world = A_obj @ R_obj
    """
    c_obj = np.array(obb_obj["center"], dtype=np.float64).reshape(3,)
    R_obj = np.array(obb_obj["axes"], dtype=np.float64)
    if R_obj.shape != (3, 3):
        raise ValueError("obb['axes'] must be (3,3)")

    c_world = origin + (A_obj @ c_obj)
    R_world = A_obj @ R_obj

    return {
        "center": c_world.tolist(),
        "axes": R_world.tolist(),
        "extents": obb_obj["extents"],
    }


# ----------------------------
# OBB -> LineSet
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


# ----------------------------
# Auto-detect whether to transform OBBs
# ----------------------------

def _extract_anchor_pair_from_change(nb_change: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns (anchor_local, anchor_world) if present, else None.
    """
    # anchor_world
    aw = None
    dbg = nb_change.get("debug", {}) or {}
    edge = dbg.get("edge", {}) or {}
    if "anchor_world" in edge:
        aw = np.array(edge["anchor_world"], dtype=np.float64).reshape(3,)

    # anchor_local
    al = None
    op = nb_change.get("op", {}) or {}
    op_dbg = op.get("debug", {}) or {}
    vol_info = op_dbg.get("volume_info", {}) or {}
    if "anchor_local" in vol_info:
        al = np.array(vol_info["anchor_local"], dtype=np.float64).reshape(3,)

    if aw is None or al is None:
        return None
    return al, aw


def _decide_transform_mode(
    constraints: Dict[str, Any],
    changes: Dict[str, Any],
    eps_ok: float = 1e-2,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (apply_object_to_world, origin, A_obj).

    apply_object_to_world=True means:
      - treat OBBs in changes.json as OBJECT space and convert to WORLD using constraints['object_space'].

    If anchor tests fail or object_space missing:
      - returns False, None, None (treat OBBs as WORLD already).
    """
    object_space = constraints.get("object_space", None)
    if object_space is None:
        print("[VIS][AUTO] constraints['object_space'] missing -> assume OBBs are WORLD (no transform).")
        return False, None, None

    origin, A_obj = _get_object_space_T(object_space)

    neighbor_changes = changes.get("neighbor_changes", {}) or {}
    best = None  # (err, nb_name)
    for nb, nb_change in neighbor_changes.items():
        pair = _extract_anchor_pair_from_change(nb_change)
        if pair is None:
            continue
        anchor_local, anchor_world = pair
        pred_world = origin + (A_obj @ anchor_local)
        err = float(np.linalg.norm(pred_world - anchor_world))
        if best is None or err < best[0]:
            best = (err, nb)

    if best is None:
        print("[VIS][AUTO] No (anchor_local, anchor_world) pairs found -> assume OBBs are WORLD (no transform).")
        return False, None, None

    err, nb = best
    print(f"[VIS][AUTO] Best anchor check neighbor='{nb}' | err_obj2world={err:.6g}")

    if err <= eps_ok:
        print("[VIS][AUTO] Anchor check PASSED -> apply OBJECT->WORLD transform to all OBBs.")
        return True, origin, A_obj

    print("[VIS][AUTO] Anchor check FAILED -> assume OBBs are already WORLD (no transform).")
    return False, None, None


def _maybe_world_obb(
    obb: Dict[str, Any],
    apply_obj2world: bool,
    origin: Optional[np.ndarray],
    A_obj: Optional[np.ndarray],
) -> Dict[str, Any]:
    if not apply_obj2world:
        return obb
    assert origin is not None and A_obj is not None
    return _obb_object_to_world(obb, origin, A_obj)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    constraints = _load_json(CONSTRAINTS_PATH)
    changes = _load_json(AEP_CHANGES_PATH)

    # Use OBBs from aep_changes.json itself (most consistent), falling back only if needed.
    target = changes.get("target", None) or (changes.get("target_edit", {}) or {}).get("target", None)
    if not target:
        raise ValueError("Could not find target label in aep_changes.json (expected 'target' or 'target_edit.target').")

    target_change = (changes.get("target_edit", {}) or {}).get("change", {}) or {}
    target_before_raw = target_change.get("before_obb", None)
    target_after_raw = target_change.get("after_obb", None)
    if target_before_raw is None or target_after_raw is None:
        raise ValueError("target_edit.change.before_obb/after_obb missing in aep_changes.json.")

    neighbor_changes = changes.get("neighbor_changes", {}) or {}
    if not neighbor_changes:
        print("[VIS] No changed neighbors -> nothing to show.")
        return

    apply_obj2world, origin, A_obj = _decide_transform_mode(constraints, changes, eps_ok=1e-2)

    overlay = _load_overlay_ply_gray(OVERLAY_PLY)

    # Precompute target boxes
    target_before = _maybe_world_obb(target_before_raw, apply_obj2world, origin, A_obj)
    target_after = _maybe_world_obb(target_after_raw, apply_obj2world, origin, A_obj)

    changed_names = list(neighbor_changes.keys())
    print(f"[VIS] target={target} | changed neighbors={len(changed_names)} | apply_obj2world={apply_obj2world}")

    for i, nb in enumerate(changed_names):
        nb_change = neighbor_changes.get(nb, {}) or {}
        nb_before_raw = nb_change.get("before_obb", None)
        nb_after_raw = nb_change.get("after_obb", None)

        if nb_before_raw is None or nb_after_raw is None:
            print(f"[VIS][WARN] neighbor '{nb}' missing before_obb/after_obb in aep_changes.json -> skip.")
            continue

        nb_before = _maybe_world_obb(nb_before_raw, apply_obj2world, origin, A_obj)
        nb_after = _maybe_world_obb(nb_after_raw, apply_obj2world, origin, A_obj)

        geoms: List[o3d.geometry.Geometry] = []
        if overlay is not None:
            geoms.append(_geom_copy(overlay))

        # ---------- TARGET (blue then red) ----------
        geoms.append(_paint_lineset(_obb_to_lineset(target_before), (0.0, 0.0, 1.0)))
        geoms.append(_paint_lineset(_obb_to_lineset(target_after), (1.0, 0.0, 0.0)))

        # ---------- NEIGHBOR (blue then red) ----------
        geoms.append(_paint_lineset(_obb_to_lineset(nb_before), (0.0, 0.0, 1.0)))
        geoms.append(_paint_lineset(_obb_to_lineset(nb_after), (1.0, 0.0, 0.0)))

        reason = nb_change.get("reason", "unknown")
        case = nb_change.get("case", None)
        tag = f"{reason}" + (f" | case={case}" if case is not None else "")

        wn = f"AEP VIS [{i+1}/{len(changed_names)}] target={target} neighbor={nb} | {tag}"
        o3d.visualization.draw_geometries(geoms, window_name=wn)


if __name__ == "__main__":
    main()
