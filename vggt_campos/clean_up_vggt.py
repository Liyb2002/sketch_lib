#!/usr/bin/env python3
import os
import json
import numpy as np
import open3d as o3d
import cv2

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "output_scene")

PLY_IN  = os.path.join(OUTPUT_DIR, "fused_model.ply")
PLY_OUT = os.path.join(OUTPUT_DIR, "fused_model_clean.ply")

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def gentle_clean_point_cloud(pcd: o3d.geometry.PointCloud):
    """
    Gentle cleanup:
      1) Remove extreme distance outliers based on distance to centroid (1% / 99.5%).
      2) Light statistical outlier removal.
    """
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        print("[WARN] Empty point cloud, skipping cleaning.")
        return pcd

    # --- Step 1: distance-based trimming (very gentle) ---
    center = pts.mean(axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    lo, hi = np.percentile(dist, [1.0, 99.5])  # keep 98.5% of points by distance

    keep_mask = (dist >= lo) & (dist <= hi)
    if not np.any(keep_mask):
        print("[WARN] Distance trimming removed all points, skipping this step.")
    else:
        pts = pts[keep_mask]
        cols = np.asarray(pcd.colors)
        if cols.shape[0] == keep_mask.shape[0]:
            cols = cols[keep_mask]
        else:
            cols = cols[keep_mask] if cols.shape[0] == np.sum(keep_mask) else cols

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)

    # --- Step 2: light statistical outlier removal ---
    # Bigger std_ratio => gentler
    if len(pcd.points) > 1000:
        pcd_clean, ind = pcd.remove_statistical_outlier(
            nb_neighbors=50,
            std_ratio=2.5
        )
        print(f"[INFO] Statistical outlier removal: {len(pcd.points)} -> {len(pcd_clean.points)} points")
        return pcd_clean

    return pcd


def render_points_from_camera(points, colors, w2c_4x4, K, save_path):
    """
    Simple point-based renderer using the same WORLD->CAM (w2c_4x4) and intrinsics K.
    Background is white, points are colored.

    points: (N, 3) in WORLD coordinates
    colors: (N, 3) in [0,1]
    """
    points = points.astype(np.float32)
    colors = colors.astype(np.float32)

    # Deduce H, W from intrinsics (cx, cy ~ image center)
    cx, cy = K[0, 2], K[1, 2]
    W = int(round(cx * 2.0))
    H = int(round(cy * 2.0))

    ones = np.ones((len(points), 1), dtype=np.float32)
    pts_hom = np.hstack([points, ones])  # (N,4)
    pts_cam = (w2c_4x4 @ pts_hom.T).T    # (N,4)

    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    # Filter points behind camera
    valid_z = z > 0.01
    if not np.any(valid_z):
        print(f"[WARN] No valid points after z>0 filter for {save_path}")
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return

    x, y, z = x[valid_z], y[valid_z], z[valid_z]
    cur_colors = colors[valid_z]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = (x * fx / z) + cx
    v = (y * fy / z) + cy

    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    valid_px = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(valid_px):
        print(f"[WARN] No valid pixels inside frame for {save_path}")
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return

    u, v, z = u[valid_px], v[valid_px], z[valid_px]
    cur_colors = cur_colors[valid_px]

    # Z-buffer style: draw far -> near so near overwrites
    sort_idx = np.argsort(-z)
    uu = u[sort_idx]
    vv = v[sort_idx]
    cc = (cur_colors[sort_idx] * 255).astype(np.uint8)

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    canvas[vv, uu] = cc

    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    if not os.path.exists(PLY_IN):
        print(f"[ERROR] Input PLY not found: {PLY_IN}")
        return

    print(f"[INFO] Loading point cloud from: {PLY_IN}")
    pcd = o3d.io.read_point_cloud(PLY_IN)
    print(f"[INFO] Loaded {len(pcd.points)} points")

    # Gentle cleanup
    pcd_clean = gentle_clean_point_cloud(pcd)

    # Save cleaned cloud
    print(f"[INFO] Saving cleaned point cloud to: {PLY_OUT}")
    o3d.io.write_point_cloud(PLY_OUT, pcd_clean)

    pts = np.asarray(pcd_clean.points)
    cols = np.asarray(pcd_clean.colors)
    if cols.shape[0] != pts.shape[0]:
        # fallback to white if colors are missing/mismatched
        cols = np.ones_like(pts, dtype=np.float32)

    # Render from all *_cam.json files in OUTPUT_DIR
    cam_files = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith("_cam.json")
    ]
    cam_files.sort()

    if not cam_files:
        print(f"[WARN] No *_cam.json files found in {OUTPUT_DIR}")
        return

    print(f"[INFO] Found {len(cam_files)} camera files.")

    for cam_file in cam_files:
        cam_path = os.path.join(OUTPUT_DIR, cam_file)
        with open(cam_path, "r") as f:
            cam_data = json.load(f)

        w2c_4x4 = np.array(cam_data["extrinsics_w2c"], dtype=np.float32)
        K = np.array(cam_data["intrinsics"], dtype=np.float32)

        base = cam_file.replace("_cam.json", "")
        out_img = os.path.join(OUTPUT_DIR, f"{base}_clean_render.png")

        print(f"[INFO] Rendering from camera: {cam_file} -> {out_img}")
        render_points_from_camera(pts, cols, w2c_4x4, K, out_img)

    print("[INFO] Done. Cleaned PLY and renders written to output_scene/")


if __name__ == "__main__":
    main()
