#!/usr/bin/env python3
"""
graph_building/pca_analysis.py

Compute one bbox per label (including unknown_x) using PCA-based OBB, with a
"rotation sanity" fix inspired by the provided snippet:

1) Compute PCA OBB from points (our fast/robust default).
2) Also compute AABB and convert it to an OBB (axis-aligned frame).
3) If PCA-OBB volume is too close to AABB volume (ratio >= 0.90), return the AABB-OBB.
   This avoids "crazy rotation" in near-isotropic / near-cubic shapes where PCA axes are unstable.

Output:
- <label_assign_dir>/label_bboxes_pca.json

Launcher stays the same: `from graph_building.pca_analysis import run as run_pca`
"""

import os
import json
import numpy as np


# If label has too few points, PCA can be unstable; we optionally add tiny jitter.
NOISE_AMT = 1e-6


def _compute_pca_obb_from_points(points: np.ndarray, noise_amt: float = NOISE_AMT):
    """
    PCA-based oriented bbox from Nx3 points.

    Returns:
      dict(center, axes, extents, volume)
    or None if insufficient points.
    """
    if points is None or points.shape[0] == 0:
        return None

    # PCA needs >= 3 non-collinear points, but we'll be conservative.
    if points.shape[0] < 3:
        # Add tiny noise points around centroid to avoid degeneracy
        center = points.mean(axis=0)
        need = 3 - points.shape[0]
        extra = center + np.random.uniform(-noise_amt, noise_amt, size=(need, 3))
        points = np.concatenate([points, extra], axis=0)

    center = points.mean(axis=0)
    X = points - center

    # Add tiny jitter to avoid singular covariance (e.g., perfectly planar/linear)
    if noise_amt > 0:
        X = X + np.random.uniform(-noise_amt, noise_amt, size=X.shape)

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # sort descending by eigenvalue
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]  # columns are principal directions

    # Ensure right-handed coordinate system (avoid reflection flips)
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0

    proj = X @ axes
    minp = proj.min(axis=0)
    maxp = proj.max(axis=0)

    extents = (maxp - minp) / 2.0
    center_local = (maxp + minp) / 2.0
    center_world = center + axes @ center_local

    volume = float((2.0 * extents[0]) * (2.0 * extents[1]) * (2.0 * extents[2]))

    return {
        "center": center_world.tolist(),
        "axes": axes.tolist(),
        "extents": extents.tolist(),
        "volume": volume,
    }


def _compute_aabb_as_obb(points: np.ndarray):
    """
    Compute axis-aligned bbox then express it as an "OBB" with identity axes.
    Returns dict(center, axes, extents, volume) or None.
    """
    if points is None or points.shape[0] == 0:
        return None

    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = (mn + mx) / 2.0
    extents = (mx - mn) / 2.0
    axes = np.eye(3, dtype=np.float64)
    volume = float((mx[0] - mn[0]) * (mx[1] - mn[1]) * (mx[2] - mn[2]))

    return {
        "center": center.tolist(),
        "axes": axes.tolist(),
        "extents": extents.tolist(),
        "volume": volume,
        "aabb_min": mn.tolist(),
        "aabb_max": mx.tolist(),
    }


def compute_pca_bbox_with_fix(points: np.ndarray, try_aabb: bool = True, vol_ratio_thresh: float = 0.90):
    """
    Inspired by the user snippet:
    - Compute PCA OBB
    - Compute AABB->OBB
    - If volumes are too similar (min(r,1/r) > thresh), use AABB-OBB to avoid unstable rotation.
    """
    obb = _compute_pca_obb_from_points(points)
    if obb is None:
        return None

    if not try_aabb:
        # remove volume key for final payload consistency
        out = {k: obb[k] for k in ("center", "axes", "extents")}
        return out

    aabb = _compute_aabb_as_obb(points)
    if aabb is None or aabb["volume"] <= 0 or obb["volume"] <= 0:
        out = {k: obb[k] for k in ("center", "axes", "extents")}
        return out

    obb_vol = obb["volume"]
    aabb_vol = aabb["volume"]

    ratio_1 = obb_vol / aabb_vol
    ratio_2 = aabb_vol / obb_vol
    stable = min(ratio_1, ratio_2) > vol_ratio_thresh

    if stable:
        # Use axis-aligned orientation when the benefit of rotation is marginal
        out = {k: aabb[k] for k in ("center", "axes", "extents")}
        return out
    else:
        out = {k: obb[k] for k in ("center", "axes", "extents")}
        return out


def run(label_assign_dir: str):
    """
    Entry point used by launcher:
      out_json = run_pca(SAVE_DIR)
    """
    ids_path = os.path.join(label_assign_dir, "assigned_label_ids.npy")
    sem_path = os.path.join(label_assign_dir, "labels_semantic.json")
    ply_path = os.path.join(label_assign_dir, "assignment_colored.ply")
    out_json = os.path.join(label_assign_dir, "label_bboxes_pca.json")

    if not os.path.isfile(ids_path):
        raise FileNotFoundError(f"Missing: {ids_path}")
    if not os.path.isfile(sem_path):
        raise FileNotFoundError(f"Missing: {sem_path}")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Missing: {ply_path}")

    assigned_ids = np.load(ids_path).reshape(-1).astype(np.int32)

    with open(sem_path, "r") as f:
        sem = json.load(f)
    label_id_to_name = {int(k): v for k, v in sem["label_id_to_name"].items()}

    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)

    if pts.shape[0] != assigned_ids.shape[0]:
        raise ValueError(f"Point count mismatch: pts={pts.shape[0]} vs assigned_ids={assigned_ids.shape[0]}")

    results = {}

    for lid, name in label_id_to_name.items():
        mask = (assigned_ids == lid)
        label_pts = pts[mask]

        # compute "fixed" PCA bbox (may fall back to AABB-OBB)
        obb_pca_fixed = compute_pca_bbox_with_fix(
            label_pts,
            try_aabb=True,
            vol_ratio_thresh=0.90
        )
        if obb_pca_fixed is None:
            continue

        results[name] = {
            "label_id": int(lid),
            "n_points": int(label_pts.shape[0]),
            "obb_pca": obb_pca_fixed,
        }

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("[PCA] saved:", out_json)
    return out_json
