#!/usr/bin/env python3
# project_aep_bbx_before_after.py
#
# Projects OBBs (before=blue, after=red) from sketch/AEP/aep_changes.json
# onto each view_{x}.png using sketch/3d_reconstruction/view_{x}_cam.json.
#
# Output:
#   sketch/back_project/view_{x}/{label}_bbx_overlay.png
#
# Important:
# - Draw BLUE first, RED second (so red is on top)
# - Everything is read ONLY from aep_changes.json (target + neighbors)

import os
import json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))

AEP_PATH  = os.path.join(ROOT, "sketch", "AEP", "aep_changes.json")
SCENE_DIR = os.path.join(ROOT, "sketch", "3d_reconstruction")
VIEWS_DIR = os.path.join(ROOT, "sketch", "views")
OUT_ROOT  = os.path.join(ROOT, "sketch", "back_project")

NUM_VIEWS = 6

BLUE = (255, 0, 0)     # BGR
RED  = (0, 0, 255)     # BGR
THICK_BLUE = 2
THICK_RED  = 3


def get_scaled_intrinsics(K_orig, src_w, src_h, target_w, target_h):
    scale_x = target_w / src_w
    scale_y = target_h / src_h
    K_new = K_orig.copy()
    K_new[0, 0] *= scale_x
    K_new[0, 2] *= scale_x
    K_new[1, 1] *= scale_y
    K_new[1, 2] *= scale_y
    return K_new


def project_world_to_px(P_world, w2c_4x4, K, H, W):
    """
    P_world: (N,3) world points
    Returns:
      px: (N,2) float
      valid: (N,) bool (z>0.01 and inside image)
    """
    N = P_world.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    Pw = np.hstack([P_world, ones])               # (N,4)
    Pc = (w2c_4x4 @ Pw.T).T                       # (N,4)

    x, y, z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid_z = z > 0.01
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    u[valid_z] = (x[valid_z] * fx / z[valid_z]) + cx
    v[valid_z] = (y[valid_z] * fy / z[valid_z]) + cy

    valid_px = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    px = np.stack([u, v], axis=1)
    return px, valid_px


def obb_corners_world(obb):
    """
    Your axes are stored as 3 vectors and used throughout your pipeline as ROWS (u0,u1,u2).
    World point: P = C + (q @ U_rows), where q is local coordinate (x,y,z).
    """
    C = np.asarray(obb["center"], dtype=np.float64)
    U = np.asarray(obb["axes"], dtype=np.float64)      # (3,3) rows u0,u1,u2
    E = np.asarray(obb["extents"], dtype=np.float64)   # half-lengths

    # 8 corners in local coords
    signs = np.array([
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ], dtype=np.float64)

    Q = signs * E[None, :]          # (8,3)
    P = C[None, :] + (Q @ U)        # (8,3)
    return P


CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]


def draw_obb(img_bgr, obb, w2c, K, color_bgr, thickness):
    H, W = img_bgr.shape[:2]
    corners = obb_corners_world(obb)
    px, valid = project_world_to_px(corners, w2c, K, H, W)
    pts = np.round(px).astype(np.int32)

    for a, b in CUBE_EDGES:
        if not (valid[a] and valid[b]):
            continue
        cv2.line(
            img_bgr,
            tuple(pts[a]),
            tuple(pts[b]),
            color_bgr,
            thickness,
            lineType=cv2.LINE_AA
        )


def load_aep_changes(aep_path):
    with open(aep_path, "r") as f:
        aep = json.load(f)

    target = aep.get("target", None)
    if not isinstance(target, str) or not target:
        raise ValueError("aep_changes.json missing 'target'")

    t_change = (aep.get("target_edit", {}) or {}).get("change", {}) or {}
    t_before = t_change.get("before_obb", None)
    t_after  = t_change.get("after_obb", None)
    if not (isinstance(t_before, dict) and isinstance(t_after, dict)):
        raise ValueError("aep_changes.json missing target_edit.change.before_obb/after_obb")

    items = []
    items.append((target, t_before, t_after))

    neigh = aep.get("neighbor_changes", {}) or {}
    if not isinstance(neigh, dict):
        neigh = {}

    for name, rec in neigh.items():
        if not isinstance(rec, dict):
            continue
        b = rec.get("before_obb", None)
        a = rec.get("after_obb", None)
        if isinstance(name, str) and isinstance(b, dict) and isinstance(a, dict):
            items.append((name, b, a))

    return items


def main():
    if not os.path.exists(AEP_PATH):
        raise FileNotFoundError(f"Missing: {AEP_PATH}")

    items = load_aep_changes(AEP_PATH)
    os.makedirs(OUT_ROOT, exist_ok=True)

    for x in range(NUM_VIEWS):
        view_name = f"view_{x}"
        img_path = os.path.join(VIEWS_DIR, f"{view_name}.png")
        cam_path = os.path.join(SCENE_DIR, f"{view_name}_cam.json")

        if not os.path.exists(img_path):
            print(f"[skip] missing image: {img_path}")
            continue
        if not os.path.exists(cam_path):
            print(f"[skip] missing cam: {cam_path}")
            continue

        img0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img0 is None:
            print(f"[skip] failed to read image: {img_path}")
            continue

        H, W = img0.shape[:2]

        with open(cam_path, "r") as f:
            cam = json.load(f)

        w2c = np.array(cam["extrinsics_w2c"], dtype=np.float64)
        K_orig = np.array(cam["intrinsics"], dtype=np.float64)

        # same fallback scaling heuristic as your 2D->3D mapping script
        src_w = float(K_orig[0, 2] * 2.0)
        src_h = float(K_orig[1, 2] * 2.0)
        K = get_scaled_intrinsics(K_orig, src_w, src_h, W, H)

        out_dir = os.path.join(OUT_ROOT, view_name)
        os.makedirs(out_dir, exist_ok=True)

        # For each changed label: draw its own overlay image (NO "overlay_all")
        for (label, obb_before, obb_after) in items:
            img = img0.copy()

            # BLUE first, RED second
            draw_obb(img, obb_before, w2c, K, BLUE, THICK_BLUE)
            draw_obb(img, obb_after,  w2c, K, RED,  THICK_RED)

            out_path = os.path.join(out_dir, f"{label}_bbx_overlay.png")
            cv2.imwrite(out_path, img)
            print(f"[ok] {view_name}: {out_path}")

    print("[done]")


if __name__ == "__main__":
    main()
