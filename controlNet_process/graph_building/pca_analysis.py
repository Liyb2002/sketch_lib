#!/usr/bin/env python3
# graph_building/pca_analysis.py

import os
import json
import numpy as np

from graph_building.object_space import world_to_object


# ----------------------------
# Helpers
# ----------------------------

def _ensure_right_handed(R: np.ndarray) -> np.ndarray:
    # R columns are axes. Ensure det(R) = +1
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R


def _fix_axis_signs_deterministic(R: np.ndarray) -> np.ndarray:
    """
    Make sign deterministic by anchoring each axis to world basis:
    the largest-magnitude component of each axis is forced to be positive.
    R columns are axes.
    """
    R2 = R.copy()
    for i in range(3):
        v = R2[:, i]
        j = int(np.argmax(np.abs(v)))
        if v[j] < 0:
            R2[:, i] *= -1.0
    R2 = _ensure_right_handed(R2)
    return R2


def _obb_half_extents_volume(extents_half: np.ndarray) -> float:
    # extents_half are half-lengths; full lengths = 2*e
    # volume = prod(full lengths) = 8 * prod(e)
    e = np.asarray(extents_half, dtype=np.float64)
    if e.shape != (3,):
        return float("inf")
    e = np.maximum(e, 0.0)
    return float(8.0 * e[0] * e[1] * e[2])


def compute_aabb_in_object_space(points_world: np.ndarray, origin: np.ndarray, axes: np.ndarray):
    """
    Compute AABB in object frame. Return OBB in world frame with:
      - axes = object axes
      - extents from object-frame AABB (half-lengths)
      - center mapped back to world
    """
    if points_world.shape[0] == 0:
        return None

    pts_local = world_to_object(points_world, origin, axes)

    mn = pts_local.min(axis=0)
    mx = pts_local.max(axis=0)

    center_local = (mn + mx) / 2.0
    extents = (mx - mn) / 2.0  # half-lengths

    # world center: origin + axes @ center_local  (axes columns)
    center_world = origin + axes @ center_local

    return {
        "center": center_world.tolist(),
        "axes": axes.tolist(),         # 3x3 (columns are axes)
        "extents": extents.tolist(),   # half-lengths
        "aabb_local_min": mn.tolist(), # debug
        "aabb_local_max": mx.tolist(),
    }


def _compute_tight_obb_open3d(points_world: np.ndarray):
    """
    Compute a tighter oriented bounding box for the component, allowing rotation.

    Returns dict with:
      center (world), axes (3x3, columns), extents (half-lengths)
    or None if not possible.
    """
    if points_world.shape[0] < 4:
        return None

    import open3d as o3d

    # Open3D expects float64 points
    pts = np.asarray(points_world, dtype=np.float64)
    try:
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(pts)
        )
    except Exception:
        return None

    c = np.asarray(obb.center, dtype=np.float64)
    R = np.asarray(obb.R, dtype=np.float64)  # rotation matrix
    # In Open3D, obb.extent is full lengths along local axes
    full = np.asarray(obb.extent, dtype=np.float64)
    if full.shape != (3,):
        return None

    # Make axes deterministic & right-handed (doesn't change the box)
    R = _fix_axis_signs_deterministic(R)

    ext_half = 0.5 * full
    return {
        "center": c.tolist(),
        "axes": R.tolist(),            # 3x3 (columns are axes)
        "extents": ext_half.tolist(),  # half-lengths
    }


def run(label_assign_dir: str, object_space: dict):
    """
    Called by launcher. Saves:
      <label_assign_dir>/label_bboxes_pca.json

    Output schema remains compatible:
      results[name]["obb_pca"] = {center, axes, extents}

    Behavior change:
      - Always compute the current object-frame AABB-lifted OBB (the old behavior).
      - Also compute a tighter per-component OBB (rotation allowed) using Open3D.
      - If the tighter OBB reduces bbox "size" (volume) by >10%, use it.
        Otherwise keep the old object-frame AABB version.
    """
    ids_path = os.path.join(label_assign_dir, "assigned_label_ids.npy")
    sem_path = os.path.join(label_assign_dir, "labels_semantic.json")
    ply_path = os.path.join(label_assign_dir, "assignment_colored.ply")
    out_json = os.path.join(label_assign_dir, "label_bboxes_pca.json")

    for p in [ids_path, sem_path, ply_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing: {p}")

    assigned_ids = np.load(ids_path).reshape(-1).astype(np.int32)

    with open(sem_path, "r") as f:
        sem = json.load(f)
    label_id_to_name = {int(k): v for k, v in sem["label_id_to_name"].items()}

    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)

    if pts.shape[0] != assigned_ids.shape[0]:
        raise ValueError(f"Point count mismatch: pts={pts.shape[0]} vs ids={assigned_ids.shape[0]}")

    origin = np.array(object_space["origin"], dtype=np.float64)
    axes = np.array(object_space["axes"], dtype=np.float64)  # 3x3 (columns are axes)
    if axes.shape != (3, 3):
        raise ValueError("object_space['axes'] must be 3x3")

    results = {}
    improve_thresh = 0.1  # >10% volume reduction

    for lid, name in label_id_to_name.items():
        mask = (assigned_ids == lid)
        label_pts = pts[mask]
        if label_pts.shape[0] == 0:
            continue

        # (A) baseline: object-space AABB lifted back as an OBB with global axes
        obb_base = compute_aabb_in_object_space(label_pts, origin, axes)
        if obb_base is None:
            continue

        base_ext = np.asarray(obb_base["extents"], dtype=np.float64)
        base_vol = _obb_half_extents_volume(base_ext)

        # (B) candidate: tighter per-component OBB with rotation allowed
        obb_tight = _compute_tight_obb_open3d(label_pts)

        chosen = "object_aabb"
        chosen_obb = {
            "center": obb_base["center"],
            "axes": obb_base["axes"],
            "extents": obb_base["extents"],
        }

        debug = {
            "method_chosen": chosen,
            "base_volume": float(base_vol),
            "tight_volume": None,
            "volume_improvement_frac": None,
            "debug_object_aabb": {
                "min": obb_base["aabb_local_min"],
                "max": obb_base["aabb_local_max"],
            },
        }

        if obb_tight is not None:
            tight_ext = np.asarray(obb_tight["extents"], dtype=np.float64)
            tight_vol = _obb_half_extents_volume(tight_ext)

            debug["tight_volume"] = float(tight_vol)

            if np.isfinite(base_vol) and base_vol > 0:
                improvement = (base_vol - tight_vol) / base_vol
                debug["volume_improvement_frac"] = float(improvement)

                # Use tight OBB only if it is >10% smaller in volume
                if improvement > improve_thresh:
                    chosen = "tight_obb"
                    chosen_obb = {
                        "center": obb_tight["center"],
                        "axes": obb_tight["axes"],
                        "extents": obb_tight["extents"],
                    }
                    debug["method_chosen"] = chosen

        results[name] = {
            "label_id": int(lid),
            "n_points": int(label_pts.shape[0]),
            "obb_pca": chosen_obb,  # keep same key name for compatibility
            "debug_bbox_choice": debug,
        }

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("[BBOX] saved:", out_json)
    return out_json
