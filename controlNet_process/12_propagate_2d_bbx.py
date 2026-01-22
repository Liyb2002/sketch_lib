#!/usr/bin/env python3
# verify_backproject_k20_scaledK.py
#
# Hard-coded back-projection verification using the SAME camera math as your 2D->3D labeling code:
# - points: sketch/3d_reconstruction/clustering_k20_points.ply
# - camera: sketch/3d_reconstruction/view_{x}_cam.json
# - target image size: sketch/views/view_{x}.png
# - K_scaled computed from (src_w,src_h) -> (target_w,target_h)
# - output: sketch/back_project/view_{x}/render_k20.png

import os
import json
import numpy as np
import cv2
import open3d as o3d

ROOT = os.path.dirname(os.path.abspath(__file__))

SCENE_DIR = os.path.join(ROOT, "sketch", "3d_reconstruction")
VIEWS_DIR = os.path.join(ROOT, "sketch", "views")
OUT_ROOT  = os.path.join(ROOT, "sketch", "back_project")

PLY_PATH  = os.path.join(SCENE_DIR, "clustering_k20_points.ply")

NUM_VIEWS = 6
OCCLUSION_THRESHOLD = 0.05


def get_scaled_intrinsics(K_orig, src_w, src_h, target_w, target_h):
    scale_x = target_w / src_w
    scale_y = target_h / src_h
    K_new = K_orig.copy()
    K_new[0, 0] *= scale_x
    K_new[0, 2] *= scale_x
    K_new[1, 1] *= scale_y
    K_new[1, 2] *= scale_y
    return K_new


def project_points(points, w2c_4x4, K, H, W):
    ones = np.ones((len(points), 1), dtype=np.float64)
    pts_hom = np.hstack([points, ones])               # (N,4)
    pts_cam = (w2c_4x4 @ pts_hom.T).T                 # (N,4)

    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid_z = z > 0.01

    u = np.zeros_like(x)
    v = np.zeros_like(y)

    u[valid_z] = (x[valid_z] * fx / z[valid_z]) + cx
    v[valid_z] = (y[valid_z] * fy / z[valid_z]) + cy

    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    valid_px = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, valid_px


def compute_depth_buffer(u, v, z, valid_mask, H, W):
    depth = np.full((H, W), np.inf, dtype=np.float32)
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = z[valid_mask]
    np.minimum.at(depth, (v_valid, u_valid), z_valid)  # nearest (min z)
    return depth


def render_points_green_overlay(base_bgr, u, v, z, valid_mask):
    H, W = base_bgr.shape[:2]

    depth = compute_depth_buffer(u, v, z, valid_mask, H, W)

    valid_idx = np.where(valid_mask)[0]
    uu = u[valid_idx]
    vv = v[valid_idx]
    zz = z[valid_idx]

    surface = depth[vv, uu]
    visible = zz <= (surface + OCCLUSION_THRESHOLD)

    uu = uu[visible]
    vv = vv[visible]

    overlay = base_bgr.copy()
    # draw points (green)
    overlay[vv, uu] = (0, 255, 0)

    # blend lightly so you can still see the image
    out = cv2.addWeighted(base_bgr, 0.75, overlay, 0.25, 0)
    return out, int(len(uu))


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    if not os.path.exists(PLY_PATH):
        raise FileNotFoundError(f"Missing: {PLY_PATH}")

    pcd = o3d.io.read_point_cloud(PLY_PATH)
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.size == 0:
        raise RuntimeError("Loaded 0 points from clustering_k20_points.ply")

    print(f"[k20] points: N={len(points)}")

    for x in range(NUM_VIEWS):
        view_name = f"view_{x}"
        img_path = os.path.join(VIEWS_DIR, f"{view_name}.png")
        cam_path = os.path.join(SCENE_DIR, f"{view_name}_cam.json")

        if not os.path.exists(img_path):
            print(f"[skip] missing image: {img_path}")
            continue
        if not os.path.exists(cam_path):
            print(f"[skip] missing camera: {cam_path}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] failed to read image: {img_path}")
            continue

        H, W = img.shape[:2]

        with open(cam_path, "r") as f:
            cam = json.load(f)

        w2c = np.array(cam["extrinsics_w2c"], dtype=np.float64)
        K_orig = np.array(cam["intrinsics"], dtype=np.float64)

        # IMPORTANT: match your pipeline’s intrinsics scaling
        # If you have a true "source render" size, use it. Otherwise use 2*cx,2*cy (your fallback).
        src_w = float(K_orig[0, 2] * 2.0)
        src_h = float(K_orig[1, 2] * 2.0)
        K = get_scaled_intrinsics(K_orig, src_w, src_h, W, H)

        u, v, z, valid = project_points(points, w2c, K, H, W)
        out, n_vis = render_points_green_overlay(img, u, v, z, valid)

        out_dir = os.path.join(OUT_ROOT, view_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "render_k20.png")
        cv2.imwrite(out_path, out)

        print(f"[ok] {view_name}: wrote {out_path} | visible_points={n_vis}")

    print("[done]")


if __name__ == "__main__":
    main()
