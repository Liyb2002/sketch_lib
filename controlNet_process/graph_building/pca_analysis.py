#!/usr/bin/env python3
# graph_building/pca_analysis.py

import os
import json
import numpy as np

from graph_building.object_space import world_to_object


def compute_aabb_in_object_space(points_world: np.ndarray, origin: np.ndarray, axes: np.ndarray):
    """
    Compute AABB in object frame. Return OBB in world frame with:
      - axes = object axes
      - extents from object-frame AABB
      - center mapped back to world
    """
    if points_world.shape[0] == 0:
        return None

    pts_local = world_to_object(points_world, origin, axes)

    mn = pts_local.min(axis=0)
    mx = pts_local.max(axis=0)

    center_local = (mn + mx) / 2.0
    extents = (mx - mn) / 2.0

    # world center: origin + axes @ center_local
    center_world = origin + axes @ center_local

    return {
        "center": center_world.tolist(),
        "axes": axes.tolist(),         # 3x3
        "extents": extents.tolist(),   # half-lengths
        "aabb_local_min": mn.tolist(), # useful debug
        "aabb_local_max": mx.tolist(),
    }


def run(label_assign_dir: str, object_space: dict):
    """
    Called by launcher. Saves:
      <label_assign_dir>/label_bboxes_pca.json

    NOTE: field name stays "obb_pca" so launcher stays simple.
    But this is actually "object-space AABB lifted back to world as an OBB".
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

    for lid, name in label_id_to_name.items():
        mask = (assigned_ids == lid)
        label_pts = pts[mask]
        if label_pts.shape[0] == 0:
            continue

        obb = compute_aabb_in_object_space(label_pts, origin, axes)
        if obb is None:
            continue

        results[name] = {
            "label_id": int(lid),
            "n_points": int(label_pts.shape[0]),
            "obb_pca": {  # keep same key name for compatibility
                "center": obb["center"],
                "axes": obb["axes"],
                "extents": obb["extents"],
            },
            "debug_object_aabb": {
                "min": obb["aabb_local_min"],
                "max": obb["aabb_local_max"],
            }
        }

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("[BBOX] saved:", out_json)
    return out_json
