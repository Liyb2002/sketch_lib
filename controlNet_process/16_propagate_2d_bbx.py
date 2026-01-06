#!/usr/bin/env python3
# 16_propagate_2d_bbx.py
"""
16_propagate_2d_bbx.py

Step 1 for 2D propagation (projection-only):
- Read sketch/target_edit/all_labels_aabbs.json
- Collect labels with is_changed == True (3D AABB before/after)
- For each view x=0..5:
    - Load sketch/3d_reconstruction/view_{x}_cam.json (intrinsics K, extrinsics_w2c)
    - Project 3D AABB corners (before & after) into 2D (pixel coords)
    - Compute 2D bbox (min/max in pixels) for before and after
    - Compute a "2D bbox change" as (scale, translate) mapping BEFORE->AFTER in *normalized* units:
        scale_x = w_after / w_before
        scale_y = h_after / h_before
        trans_x = (cx_after - cx_before) / w_before
        trans_y = (cy_after - cy_before) / h_before
      This matches your "assume original 2D bbox is 1x1" (treat before bbox as unit size).
    - Save per-view JSON to:
        sketch/target_edit/2d_projection/view_{x}/changed_labels_2d_bbox.json

Notes:
- This step does NOT touch 2D segmentations yet; it only tells how 3D bbox edits would look in 2D.
- If a bbox projects partially behind the camera (Z<=0), we mark it invalid for that view.
"""

import os
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


# ------------------------ IO ------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------ Geometry ------------------------

def corners_of_aabb(mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    """
    Return (8,3) corners of axis-aligned bounding box.
    """
    x0, y0, z0 = mn.tolist()
    x1, y1, z1 = mx.tolist()
    return np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float64)


def project_points_w2c(K: np.ndarray, W2C: np.ndarray, pts_world: np.ndarray, z_eps: float = 1e-9
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project world points to image pixels.
    Args:
      K: (3,3)
      W2C: (4,4) world-to-camera extrinsics
      pts_world: (N,3)
    Returns:
      pts_2d: (N,2) pixels
      valid:  (N,) boolean, True where Z>0
    """
    N = pts_world.shape[0]
    pts_h = np.concatenate([pts_world, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N,4)
    cam_h = (W2C @ pts_h.T).T  # (N,4)
    Xc = cam_h[:, 0]
    Yc = cam_h[:, 1]
    Zc = cam_h[:, 2]

    valid = Zc > z_eps
    Z_safe = np.where(valid, Zc, 1.0)  # avoid div-by-zero; invalid will be flagged

    # Apply intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * (Xc / Z_safe) + cx
    v = fy * (Yc / Z_safe) + cy
    pts_2d = np.stack([u, v], axis=1)
    return pts_2d, valid


def bbox2d_from_points(pts_2d: np.ndarray, valid: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    """
    Returns (min_xy, max_xy, is_valid_bbox).
    If fewer than 1 valid point, returns (None, None, False).
    """
    if valid is None or valid.sum() == 0:
        return None, None, False
    p = pts_2d[valid]
    mn = np.min(p, axis=0)
    mx = np.max(p, axis=0)
    return mn, mx, True


def bbox_change_normalized(before_mn: np.ndarray, before_mx: np.ndarray,
                           after_mn: np.ndarray, after_mx: np.ndarray,
                           eps: float = 1e-9) -> Dict[str, float]:
    """
    Compute scale+translation from before bbox -> after bbox in normalized units where
    w_before and h_before are treated as 1.

    Returns dict:
      scale_x, scale_y, trans_x, trans_y
    """
    wb = float(before_mx[0] - before_mn[0])
    hb = float(before_mx[1] - before_mn[1])
    wa = float(after_mx[0] - after_mn[0])
    ha = float(after_mx[1] - after_mn[1])

    # Guard against degenerate before boxes
    wb_safe = wb if abs(wb) > eps else 1.0
    hb_safe = hb if abs(hb) > eps else 1.0

    cbx = float(0.5 * (before_mx[0] + before_mn[0]))
    cby = float(0.5 * (before_mx[1] + before_mn[1]))
    cax = float(0.5 * (after_mx[0] + after_mn[0]))
    cay = float(0.5 * (after_mx[1] + after_mn[1]))

    scale_x = wa / wb_safe
    scale_y = ha / hb_safe
    trans_x = (cax - cbx) / wb_safe
    trans_y = (cay - cby) / hb_safe

    return {
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "trans_x": float(trans_x),
        "trans_y": float(trans_y),
    }


# ------------------------ Parse all_labels_aabbs ------------------------

def np3(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)


def get_changed_labels_3d(data: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns:
      changed[label] = {
        before_min, before_max, after_min, after_max
      }
    """
    labels = data.get("labels", {})
    if not isinstance(labels, dict):
        raise ValueError("all_labels_aabbs.json missing 'labels' dict")

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for label, entry in labels.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("is_changed", False) is True:
            b = entry.get("before", None)
            a = entry.get("after", None)
            if not (isinstance(b, dict) and isinstance(a, dict)):
                continue
            out[label] = {
                "before_min": np3(b["min"]),
                "before_max": np3(b["max"]),
                "after_min": np3(a["min"]),
                "after_max": np3(a["max"]),
            }
    return out


# ------------------------ Main ------------------------

def process_view(cam_path: str, changed_3d: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    cam = load_json(cam_path)
    K = np.asarray(cam["intrinsics"], dtype=np.float64)
    W2C = np.asarray(cam["extrinsics_w2c"], dtype=np.float64)

    per_label: Dict[str, Any] = {}

    for label, aabbs in changed_3d.items():
        bmn = aabbs["before_min"]
        bmx = aabbs["before_max"]
        amn = aabbs["after_min"]
        amx = aabbs["after_max"]

        # Project BEFORE
        b_corners = corners_of_aabb(bmn, bmx)
        b_pts2d, b_valid = project_points_w2c(K, W2C, b_corners)
        b2_mn, b2_mx, b_ok = bbox2d_from_points(b_pts2d, b_valid)

        # Project AFTER
        a_corners = corners_of_aabb(amn, amx)
        a_pts2d, a_valid = project_points_w2c(K, W2C, a_corners)
        a2_mn, a2_mx, a_ok = bbox2d_from_points(a_pts2d, a_valid)

        ok = bool(b_ok and a_ok)

        entry: Dict[str, Any] = {
            "valid": ok,
            # Save pixel bboxes for debugging / later use:
            "before_2d_bbox_px": None if not b_ok else {"min": b2_mn.tolist(), "max": b2_mx.tolist()},
            "after_2d_bbox_px": None if not a_ok else {"min": a2_mn.tolist(), "max": a2_mx.tolist()},
        }

        if ok:
            entry["change_from_before_to_after_normalized"] = bbox_change_normalized(b2_mn, b2_mx, a2_mn, a2_mx)
        else:
            entry["change_from_before_to_after_normalized"] = None

        per_label[label] = entry

    return {
        "camera_path": cam_path,
        "labels": per_label,
    }


def main(
    all_labels_path: str,
    cam_dir: str,
    out_dir: str,
    num_views: int = 6
):
    data = load_json(all_labels_path)
    changed_3d = get_changed_labels_3d(data)

    os.makedirs(out_dir, exist_ok=True)

    for vid in range(num_views):
        cam_path = os.path.join(cam_dir, f"view_{vid}_cam.json")
        if not os.path.isfile(cam_path):
            raise FileNotFoundError(f"Missing camera json: {cam_path}")

        view_out = process_view(cam_path, changed_3d)

        view_folder = os.path.join(out_dir, f"view_{vid}")
        os.makedirs(view_folder, exist_ok=True)
        out_path = os.path.join(view_folder, "changed_labels_2d_bbox.json")
        save_json(out_path, view_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all_labels",
        default="sketch/target_edit/all_labels_aabbs.json",
        help="Path to all_labels_aabbs.json",
    )
    parser.add_argument(
        "--cam_dir",
        default="sketch/3d_reconstruction",
        help="Folder containing view_{x}_cam.json",
    )
    parser.add_argument(
        "--out_dir",
        default="sketch/target_edit/2d_projection",
        help="Output folder: will create view_{x}/changed_labels_2d_bbox.json",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=6,
        help="Number of views (expects view_0..view_{num_views-1})",
    )
    args = parser.parse_args()

    main(args.all_labels, args.cam_dir, args.out_dir, args.num_views)
