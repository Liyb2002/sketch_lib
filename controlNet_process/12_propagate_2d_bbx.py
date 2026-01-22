#!/usr/bin/env python3
# project_aep_bbx_per_label_blue_then_red.py
#
# Reads sketch/AEP/aep_changes.json and projects per-label OBB overlays onto each sketch/views/view_{x}.png.
# Output: sketch/back_project/view_{x}/
#
# For each label:
# - draw BEFORE (blue) first
# - draw AFTER  (red) second (so it overlays blue)
#
# Note:
# - target has before_obb + after_obb in aep_changes.json
# - neighbors in neighbor_changes only have after_obb (no true "before" in this file)

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

# colors (BGR)
BLUE = (255, 0, 0)
RED  = (0, 0, 255)

THICK_BLUE = 2
THICK_RED  = 2


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
    N = P_world.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    Pw = np.hstack([P_world, ones])                         # (N,4)
    Pc = (w2c_4x4 @ Pw.T).T                                 # (N,4)

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
    obb: {center:[3], axes:[[3],[3],[3]] rows u0,u1,u2, extents:[3] half-lengths}
    """
    C = np.asarray(obb["center"], dtype=np.float64)
    U = np.asarray(obb["axes"], dtype=np.float64)      # rows u0,u1,u2
    E = np.asarray(obb["extents"], dtype=np.float64)   # half-lengths

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

    Q = signs * E[None, :]
    corners = C[None, :] + (Q @ U)   # sum(q_i * u_i)
    return corners


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

    drawn = 0
    for a, b in CUBE_EDGES:
        if not (valid[a] and valid[b]):
            continue
        cv2.line(img_bgr, tuple(pts[a]), tuple(pts[b]), color_bgr, thickness, lineType=cv2.LINE_AA)
        drawn += 1
    return drawn


def main():
    if not os.path.exists(AEP_PATH):
        raise FileNotFoundError(f"Missing: {AEP_PATH}")

    with open(AEP_PATH, "r") as f:
        aep = json.load(f)

    target = aep.get("target", None)
    target_change = (aep.get("target_edit", {}) or {}).get("change", {}) or {}
    t_before = target_change.get("before_obb", None)
    t_after  = target_change.get("after_obb", None)

    if not target or not isinstance(t_before, dict) or not isinstance(t_after, dict):
        raise ValueError("aep_changes.json missing target or target_edit.change.before_obb/after_obb")

    neighbor_changes = aep.get("neighbor_changes", {}) or {}
    neighbor_names = sorted(list(neighbor_changes.keys()))

    os.makedirs(OUT_ROOT, exist_ok=True)

    # prep per-label OBB sets
    # target has before+after; neighbors only have after in this file
    labels = [(target, t_before, t_after, True)]
    for nb in neighbor_names:
        nb_after = (neighbor_changes[nb].get("after_obb", None) or {})
        if isinstance(nb_after, dict) and "center" in nb_after:
            labels.append((nb, None, nb_after, False))

    for x in range(NUM_VIEWS):
        view_name = f"view_{x}"
        img_path = os.path.join(VIEWS_DIR, f"{view_name}.png")
        cam_path = os.path.join(SCENE_DIR, f"{view_name}_cam.json")

        if not os.path.exists(img_path) or not os.path.exists(cam_path):
            continue

        img0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img0 is None:
            continue
        H, W = img0.shape[:2]

        with open(cam_path, "r") as f:
            cam = json.load(f)

        w2c = np.array(cam["extrinsics_w2c"], dtype=np.float64)
        K_orig = np.array(cam["intrinsics"], dtype=np.float64)

        # same fallback as your pipeline
        src_w = float(K_orig[0, 2] * 2.0)
        src_h = float(K_orig[1, 2] * 2.0)
        K = get_scaled_intrinsics(K_orig, src_w, src_h, W, H)

        out_dir = os.path.join(OUT_ROOT, view_name)
        os.makedirs(out_dir, exist_ok=True)

        for (name, before_obb, after_obb, has_before) in labels:
            img = img0.copy()

            # blue first, red second
            if has_before and before_obb is not None:
                draw_obb(img, before_obb, w2c, K, BLUE, THICK_BLUE)

            draw_obb(img, after_obb, w2c, K, RED, THICK_RED)

            out_path = os.path.join(out_dir, f"bbx_{name}.png")
            cv2.imwrite(out_path, img)

        print(f"[ok] {view_name}: wrote {len(labels)} per-label bbx overlays to {out_dir}")

    if len(neighbor_names) > 0:
        print("[note] neighbors in aep_changes.json only include after_obb; blue(before) is only drawn for the target.")
        print("       If you want neighbors blue+red too, load neighbor 'before' from the original constraints file.")

    print("[done]")


if __name__ == "__main__":
    main()
