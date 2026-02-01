#!/usr/bin/env python3
# graph_building/pca_analysis.py

import os
import json
import numpy as np

from graph_building.object_space import world_to_object


# ----------------------------
# Helpers
# ----------------------------


def _robust_outlier_mask_local(pts_local: np.ndarray, max_frac: float = 0.02) -> np.ndarray:
    """
    Return boolean mask of inliers (True = keep), removing up to max_frac points
    with largest robust distance from the median in LOCAL coordinates.

    Deterministic, and guaranteed to remove <= max_frac of points.
    """
    n = int(pts_local.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)

    # robust center/scale per axis
    med = np.median(pts_local, axis=0)
    mad = np.median(np.abs(pts_local - med), axis=0)

    # Avoid divide-by-zero: if axis has no spread, keep it from dominating
    mad = np.maximum(mad, 1e-12)

    z = (pts_local - med) / mad
    # robust distance (L2 in robust-z space)
    dist = np.sqrt((z * z).sum(axis=1))

    k_remove = int(np.floor(max_frac * n))
    if k_remove <= 0:
        return np.ones((n,), dtype=bool)

    # remove the k largest distances
    idx = np.argsort(dist)  # ascending
    keep = np.ones((n,), dtype=bool)
    keep[idx[-k_remove:]] = False
    return keep


def _compute_aabb_in_object_space_trimmed(points_world: np.ndarray,
                                         origin: np.ndarray,
                                         axes: np.ndarray,
                                         max_outlier_frac: float = 0.02):
    """
    Compute baseline AABB-in-frame OBB, then try dropping up to max_outlier_frac outliers
    (in that frame) and recompute. Return (obb_full, obb_trimmed_or_None, frac_removed).
    """
    if points_world.shape[0] == 0:
        return None, None, 0.0

    pts_local = world_to_object(points_world, origin, axes)

    keep = _robust_outlier_mask_local(pts_local, max_frac=max_outlier_frac)
    frac_removed = float(1.0 - (keep.mean() if keep.size else 1.0))

    obb_full = compute_aabb_in_object_space(points_world, origin, axes)
    obb_trim = None

    if keep.sum() >= 4 and keep.sum() < points_world.shape[0]:
        obb_trim = compute_aabb_in_object_space(points_world[keep], origin, axes)

    return obb_full, obb_trim, frac_removed


def _maybe_use_trimmed_bbox(obb_full: dict,
                            obb_trim: dict,
                            frac_removed: float,
                            improve_thresh: float = 0.10):
    """
    Decide whether to replace obb_full by obb_trim based on:
      - obb_trim exists
      - frac_removed <= 2% (already enforced by how we build it)
      - volume shrinks by > improve_thresh
    Returns (chosen_obb, debug_dict)
    """
    dbg = {
        "used_trimmed": False,
        "frac_removed": float(frac_removed),
        "full_volume": None,
        "trim_volume": None,
        "trim_improvement_frac": None,
    }
    if obb_full is None:
        return None, dbg

    full_ext = np.asarray(obb_full["extents"], dtype=np.float64)
    full_vol = _obb_half_extents_volume(full_ext)
    dbg["full_volume"] = float(full_vol)

    if obb_trim is None:
        return obb_full, dbg

    trim_ext = np.asarray(obb_trim["extents"], dtype=np.float64)
    trim_vol = _obb_half_extents_volume(trim_ext)
    dbg["trim_volume"] = float(trim_vol)

    if np.isfinite(full_vol) and full_vol > 0:
        imp = (full_vol - trim_vol) / full_vol
        dbg["trim_improvement_frac"] = float(imp)
        if imp > improve_thresh:
            dbg["used_trimmed"] = True
            return obb_trim, dbg

    return obb_full, dbg


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


def _orthonormalize_axes_columns(R: np.ndarray) -> np.ndarray:
    """
    Orthonormalize a 3x3 matrix whose columns are intended axes.
    Uses SVD projection to closest proper rotation.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("axes must be 3x3")
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


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


# ----------------------------
# New constrained "tight OBB"
# ----------------------------

def _rotation_about_unit_axis(axis_world: np.ndarray, theta: float) -> np.ndarray:
    """
    Rodrigues rotation matrix for rotating around a unit axis in world space.
    """
    a = np.asarray(axis_world, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    a = a / n

    x, y, z = float(a[0]), float(a[1]), float(a[2])
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    C = 1.0 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)


def _compute_tight_obb_constrained(points_world: np.ndarray,
                                  object_axes_cols: np.ndarray,
                                  n_steps: int = 180):
    """
    Constrained tight OBB search:
      - Start from object-space axes (3 columns).
      - Keep ONE object axis fixed (exactly aligned with object space).
      - Rotate the other two axes around that fixed axis (i.e., rotate within the perpendicular plane).
      - Try all three choices of which axis is fixed, and all angles.
      - Pick the smallest-volume box (by AABB in that rotated frame).

    Returns dict with {center, axes, extents} (half-lengths), or None.
    """
    if points_world.shape[0] < 4:
        return None

    A = _orthonormalize_axes_columns(object_axes_cols)
    # Deterministic sign so results don't flip randomly
    A = _fix_axis_signs_deterministic(A)

    # Use centroid as origin for stable numerics; this does NOT affect tightness (AABB extents are translation-invariant)
    origin = np.asarray(points_world, dtype=np.float64).mean(axis=0)

    best = None
    best_vol = float("inf")
    best_meta = None

    # Try: fix axis i, rotate around it
    for fix_i in (0, 1, 2):
        axis_fixed = A[:, fix_i]  # world direction

        for k in range(n_steps):
            theta = (2.0 * np.pi) * (k / float(n_steps))
            Rax = _rotation_about_unit_axis(axis_fixed, theta)

            # Rotate the whole frame around axis_fixed in world space
            # This keeps the fixed axis aligned (since rotating around itself doesn't move it).
            axes_try = Rax @ A

            # Clean up numerics and make deterministic
            axes_try = _orthonormalize_axes_columns(axes_try)
            axes_try = _fix_axis_signs_deterministic(axes_try)

            obb = compute_aabb_in_object_space(points_world, origin, axes_try)
            if obb is None:
                continue

            ext = np.asarray(obb["extents"], dtype=np.float64)
            vol = _obb_half_extents_volume(ext)

            if vol < best_vol:
                best_vol = vol
                best = {
                    "center": obb["center"],
                    "axes": obb["axes"],
                    "extents": obb["extents"],
                }
                best_meta = {"fixed_axis_index": int(fix_i), "theta": float(theta)}

    if best is None:
        return None

    # Attach a little debug if you want it later (kept separate from schema)
    best["_debug_constrained"] = best_meta
    return best


def run(label_assign_dir: str, object_space: dict):
    """
    Called by launcher. Saves:
      <label_assign_dir>/label_bboxes_pca.json

    Output schema remains compatible:
      results[name]["obb_pca"] = {center, axes, extents}

    Behavior:
      - Always compute the current object-frame AABB-lifted OBB (unchanged).
      - Also compute a constrained tight OBB:
          * start from object_space axes
          * keep one object axis fixed
          * rotate other two around it
          * pick smallest-volume AABB-in-frame
      - If constrained tight OBB reduces bbox volume by >10%, use it.
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

    # Make sure object-space axes form a proper rigid frame (important for tightness!)
    axes = _orthonormalize_axes_columns(axes)
    axes = _fix_axis_signs_deterministic(axes)

    results = {}
    improve_thresh = 0.10  # require >10% volume reduction to switch

    # More steps = tighter but slower; 180 is usually fine. Use 360 if you want.
    n_steps = 180

    for lid, name in label_id_to_name.items():
        mask = (assigned_ids == lid)
        label_pts = pts[mask]
        if label_pts.shape[0] == 0:
            continue

        # (A) baseline: object-space AABB lifted back as an OBB with global object axes (UNCHANGED)
        obb_base_full, obb_base_trim, frac_removed_base = _compute_aabb_in_object_space_trimmed(
            label_pts, origin, axes, max_outlier_frac=0.02
        )
        if obb_base_full is None:
            continue

        obb_base, dbg_base_trim = _maybe_use_trimmed_bbox(
            obb_base_full, obb_base_trim, frac_removed_base, improve_thresh=0.10
        )

        base_ext = np.asarray(obb_base["extents"], dtype=np.float64)
        base_vol = _obb_half_extents_volume(base_ext)

        # (B) candidate: constrained tight OBB (2 axes rotate in a plane, 1 axis fixed to object space)
        obb_tight = _compute_tight_obb_constrained(label_pts, axes, n_steps=n_steps)
        

        dbg_tight_trim = None
        obb_tight_chosen = obb_tight

        if obb_tight is not None:
            # Use the tight axes as the frame; origin can be anything (extents translation-invariant),
            # but for consistency use the same origin used by compute_aabb_in_object_space.
            # Here: pick centroid for stability.
            origin_t = np.asarray(label_pts, dtype=np.float64).mean(axis=0)
            axes_t = np.asarray(obb_tight["axes"], dtype=np.float64)

            obb_t_full, obb_t_trim, frac_removed_t = _compute_aabb_in_object_space_trimmed(
                label_pts, origin_t, axes_t, max_outlier_frac=0.02
            )
            # Important: obb_t_full here should match the extents you'd get in that same frame;
            # but we still want to keep the original tight center/axes schema.
            # So we swap only if trimmed improves volume >10%.
            obb_t_candidate, dbg_tight_trim = _maybe_use_trimmed_bbox(
                obb_t_full, obb_t_trim, frac_removed_t, improve_thresh=0.10
            )

            if dbg_tight_trim["used_trimmed"]:
                # Keep axes = tight axes, but use trimmed center/extents from recompute in that frame.
                obb_tight_chosen = {
                    "center": obb_t_candidate["center"],
                    "axes": obb_t_candidate["axes"],
                    "extents": obb_t_candidate["extents"],
                    "_debug_constrained": obb_tight.get("_debug_constrained", None),
                }



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
            "tight_debug": None,
        }

        if obb_tight is not None:
            tight_ext = np.asarray(obb_tight_chosen["extents"], dtype=np.float64)
            tight_vol = _obb_half_extents_volume(tight_ext)

            debug["tight_volume"] = float(tight_vol)
            debug["tight_debug"] = obb_tight.get("_debug_constrained", None)

            if np.isfinite(base_vol) and base_vol > 0:
                improvement = (base_vol - tight_vol) / base_vol
                debug["volume_improvement_frac"] = float(improvement)

                # Use tight OBB only if it is >10% smaller in volume
                if improvement > improve_thresh:
                    chosen = "tight_obb_constrained"
                    chosen_obb = {
                        "center": obb_tight_chosen["center"],
                        "axes": obb_tight_chosen["axes"],
                        "extents": obb_tight_chosen["extents"],
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
