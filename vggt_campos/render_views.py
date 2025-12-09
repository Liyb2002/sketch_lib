#!/usr/bin/env python3
"""
render_vggt_reprojections.py

Given:
  - vggt_scene/extrinsics.npy  (N, 3, 4)  # camera-from-world, OpenCV convention
  - vggt_scene/intrinsics.npy  (N, 3, 3)
  - vggt_scene/point_cloud_object_only_10k.ply  (world-space points)
  - view/view_0.png ... view/view_5.png  (original input images)

This script reprojects the 3D point cloud into each camera and renders a
binary silhouette for each view:

  vggt_scene/reproj_view_0.png
  ...
  vggt_scene/reproj_view_{N-1}.png

Each silhouette:
  - white background (255)
  - black object pixels (0)

You can overlay these on the original sketches to see how well the
inferred cameras line up.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import open3d as o3d

# ----------------- hard-coded paths -----------------

SCENE_DIR       = Path("vggt_scene")
INPUT_VIEWS_DIR = Path("view")

EXTRINSICS_PATH = SCENE_DIR / "extrinsics.npy"
INTRINSICS_PATH = SCENE_DIR / "intrinsics.npy"
POINTCLOUD_PATH = SCENE_DIR / "point_cloud_object_only_10k.ply"

OUTPUT_DIR      = SCENE_DIR  # save images next to the scene files


def load_cameras():
    if not EXTRINSICS_PATH.is_file():
        raise FileNotFoundError(f"Missing {EXTRINSICS_PATH}")
    if not INTRINSICS_PATH.is_file():
        raise FileNotFoundError(f"Missing {INTRINSICS_PATH}")

    extrinsics = np.load(EXTRINSICS_PATH)  # (N, 3, 4)
    intrinsics = np.load(INTRINSICS_PATH)  # (N, 3, 3)

    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError(f"extrinsics.npy has unexpected shape {extrinsics.shape}")
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"intrinsics.npy has unexpected shape {intrinsics.shape}")

    if extrinsics.shape[0] != intrinsics.shape[0]:
        raise ValueError(
            f"Number of extrinsics ({extrinsics.shape[0]}) != "
            f"number of intrinsics ({intrinsics.shape[0]})"
        )

    print(f"[INFO] Loaded cameras for {extrinsics.shape[0]} views.")
    return extrinsics, intrinsics


def load_point_cloud():
    if not POINTCLOUD_PATH.is_file():
        raise FileNotFoundError(f"Missing point cloud: {POINTCLOUD_PATH}")

    pcd = o3d.io.read_point_cloud(str(POINTCLOUD_PATH))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.size == 0:
        raise RuntimeError("Point cloud has zero points.")

    print(f"[INFO] Loaded point cloud: {pts.shape[0]} points from {POINTCLOUD_PATH.name}")
    return pts


def get_image_size():
    """
    Infer the image size (H, W) from view_0.png in INPUT_VIEWS_DIR.
    This should match the size used when running VGGT.
    """
    candidates = sorted(INPUT_VIEWS_DIR.glob("view_*.png"))
    if not candidates:
        raise FileNotFoundError(f"No view_*.png found in {INPUT_VIEWS_DIR}")

    img0 = Image.open(candidates[0]).convert("RGB")
    W, H = img0.size
    print(f"[INFO] Using image size H={H}, W={W} (from {candidates[0].name})")
    return H, W


def project_points_to_image(
    points_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    H: int,
    W: int,
):
    """
    Project world-space points into an image using OpenCV-style camera matrices.

    extrinsic: (3, 4) camera-from-world: X_cam = R X_world + t
    intrinsic: (3, 3)
    Returns a binary mask (H, W) with z-buffering:
      True where at least one 3D point projects, False otherwise.
    """
    R = extrinsic[:, :3]  # (3,3)
    t = extrinsic[:, 3]   # (3,)

    # World → camera
    # points_world: (N, 3)
    Xw = points_world.T   # (3, N)
    Xc = R @ Xw + t[:, None]  # (3, N)

    x = Xc[0, :]
    y = Xc[1, :]
    z = Xc[2, :]

    # Keep only points in front of the camera
    eps = 1e-6
    valid = z > eps
    if not np.any(valid):
        # No visible points
        return np.zeros((H, W), dtype=bool)

    x = x[valid]
    y = y[valid]
    z = z[valid]

    # Intrinsic projection
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    # Pixel coords
    u_pix = np.round(u).astype(np.int32)
    v_pix = np.round(v).astype(np.int32)

    # In-bounds mask
    in_bounds = (
        (u_pix >= 0) & (u_pix < W) &
        (v_pix >= 0) & (v_pix < H)
    )
    if not np.any(in_bounds):
        return np.zeros((H, W), dtype=bool)

    u_pix = u_pix[in_bounds]
    v_pix = v_pix[in_bounds]
    z = z[in_bounds]

    # Z-buffer: keep the closest point per pixel
    depth_img = np.full((H, W), np.inf, dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)

    for uu, vv, zz in zip(u_pix, v_pix, z):
        if zz < depth_img[vv, uu]:
            depth_img[vv, uu] = zz
            mask[vv, uu] = True

    return mask


def main():
    extrinsics, intrinsics = load_cameras()
    points_world = load_point_cloud()
    H, W = get_image_size()

    num_views = extrinsics.shape[0]

    for i in range(num_views):
        print(f"[INFO] Reprojecting view {i} ...")
        E = extrinsics[i]  # (3,4)
        K = intrinsics[i]  # (3,3)

        mask = project_points_to_image(points_world, E, K, H, W)

        # Render as white background, black object
        img = np.ones((H, W), dtype=np.uint8) * 255
        img[mask] = 0

        out_path = OUTPUT_DIR / f"reproj_view_{i}.png"
        Image.fromarray(img).save(out_path)
        print(f"  -> saved {out_path}")

    print("\n[DONE] Reprojected views saved in", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
