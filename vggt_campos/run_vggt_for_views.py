#!/usr/bin/env python3
"""
run_vggt_for_views.py

Multi-view VGGT pipeline with object-only 3D reconstruction.

Assumes folder layout:

    vggt_campos/
      vggt/                    # VGGT repo (with inner vggt/ package)
      view/
        view_0.png
        view_1.png
        ...
      run_vggt_for_views.py    # this script

What it does:

1) Copies view_*.png into:
       vggt_scene/images/

2) Runs VGGT-1B on all views:
   - estimates camera intrinsics & extrinsics
   - predicts per-view depth maps

3) Saves:
       vggt_scene/extrinsics.npy   (N, 3, 4)
       vggt_scene/intrinsics.npy   (N, 3, 3)
       vggt_scene/depth_maps.npy   (N, H, W)

4) Builds a 3D point cloud for the object ONLY:
   - loads original RGB images
   - resizes them to (H, W)
   - computes a brightness-based mask to keep only the central object
   - zeroes depth outside the mask
   - calls unproject_depth_map_to_point_map(...)
   - merges all views into a single point cloud

5) Saves:
       vggt_scene/point_cloud_object_only.ply
"""

import os
import sys
from pathlib import Path
import shutil

import numpy as np
import torch
from PIL import Image
import open3d as o3d

# ----------------------------------------------------------------------
# Make local VGGT repo importable.
# Layout:
#   vggt_campos/
#       vggt/   <- cloned VGGT repo (contains inner vggt package)
#       run_vggt_for_views.py
# ----------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VGGT_REPO_ROOT = os.path.join(THIS_DIR, "vggt")
sys.path.insert(0, VGGT_REPO_ROOT)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


# ---------- hard-coded paths ----------
INPUT_VIEWS_DIR = Path("view")           # expects view_0.png ... view_5.png
SCENE_DIR = Path("vggt_scene")           # output folder
IMAGES_DIR = SCENE_DIR / "images"

BG_THRESHOLD = 0.95  # pixels brighter than this (in [0,1]) are treated as background

# Max number of points to keep in final point cloud
MAX_POINTS = 500_000


def normalize_depth_tensor_to_nhw(depth_tensor: torch.Tensor) -> np.ndarray:
    """
    depth_tensor: torch tensor from VGGT depth_head, after squeezing batch dim.
      It can have one of several shapes depending on version:
        (N, 1, H, W)
        (N, H, W, 1)
        (N, 1, 1, H, W)  [before squeezing]
    We return a numpy array of shape (N, H, W).
    """
    d = depth_tensor  # torch
    depth_np = d.detach().cpu().numpy()

    # Case 1: (N, 1, H, W)
    if depth_np.ndim == 4 and depth_np.shape[1] == 1:
        depth_np = depth_np[:, 0]  # (N, H, W)
    # Case 2: (N, H, W, 1)
    elif depth_np.ndim == 4 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]  # (N, H, W)
    # Case 3: already (N, H, W)
    elif depth_np.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected depth tensor shape: {depth_np.shape}")

    return depth_np  # (N, H, W)


def build_object_masks(view_paths, target_h, target_w):
    """
    For each input RGB view, build an object mask of shape (H, W) where:
        True  = object (chair)
        False = background

    We treat bright pixels (close to white) as background.

    Returns: masks: np.ndarray of shape (N, H, W), dtype=bool
    """
    masks = []
    for p in view_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((target_w, target_h), Image.BILINEAR)
        rgb = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)

        gray = rgb.mean(axis=2)  # (H, W)
        mask = gray < BG_THRESHOLD  # object is darker than near-white bg

        # Optional: dilate slightly to include edges (if cv2 is available)
        try:
            import cv2
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        except ImportError:
            pass

        masks.append(mask)

    masks = np.stack(masks, axis=0)  # (N, H, W)
    return masks


def main():
    # 1) Prepare scene folder and copy the views
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    view_paths = sorted(
        p for p in INPUT_VIEWS_DIR.glob("view_*.png")
        if p.is_file()
    )
    if len(view_paths) == 0:
        raise RuntimeError(f"No images found in {INPUT_VIEWS_DIR} matching 'view_*.png'.")

    print("Found input views:")
    for p in view_paths:
        print("   ", p)

    for src in view_paths:
        dst = IMAGES_DIR / src.name
        shutil.copy2(src, dst)

    print(f"Copied views into {IMAGES_DIR}")

    # 2) Load images with VGGT helper (preprocessed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA not available, this will be slow on CPU.")

    major_cc = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    dtype = torch.bfloat16 if major_cc >= 8 else torch.float16

    image_names = [str(p) for p in sorted(IMAGES_DIR.glob("*.png"))]
    if len(image_names) == 0:
        raise RuntimeError(f"No PNG images found in {IMAGES_DIR}")

    print("Loading and preprocessing images for VGGT ...")
    images = load_and_preprocess_images(image_names).to(device)  # (N, 3, H, W)
    num_views = images.shape[0]
    H, W = images.shape[-2:]
    print(f"Loaded {num_views} images for VGGT, size = {H}x{W}")

    # 3) Load VGGT model
    print("Loading VGGT model (facebook/VGGT-1B) ...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    # 4) Run aggregator + camera head + depth head
    print("Running VGGT aggregator / camera_head / depth_head ...")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        images_batched = images[None]  # (1, N, 3, H, W)

        aggregated_tokens_list, ps_idx = model.aggregator(images_batched)

        # Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images.shape[-2:]
        )  # extrinsic: (1, N, 3, 4), intrinsic: (1, N, 3, 3)

        # Depth maps
        depth_map, depth_conf = model.depth_head(
            aggregated_tokens_list, images_batched, ps_idx
        )  # typically (1, N, 1, H, W) or (1, N, H, W, 1)

    # Remove batch dim
    extrinsic = extrinsic.squeeze(0)      # (N, 3, 4) torch
    intrinsic = intrinsic.squeeze(0)      # (N, 3, 3) torch
    depth_map = depth_map.squeeze(0)      # (N, ..., ..., ...) torch

    extrinsic_np = extrinsic.detach().cpu().numpy()   # (N, 3, 4)
    intrinsic_np = intrinsic.detach().cpu().numpy()   # (N, 3, 3)
    depth_np = normalize_depth_tensor_to_nhw(depth_map)  # (N, H, W)

    # Save raw cameras + depth
    np.save(SCENE_DIR / "extrinsics.npy", extrinsic_np)
    np.save(SCENE_DIR / "intrinsics.npy", intrinsic_np)
    np.save(SCENE_DIR / "depth_maps.npy", depth_np)
    print(f"Saved extrinsics, intrinsics, and depth_maps to {SCENE_DIR}")

    # 5) Build object masks from original RGB images
    print("Building object masks from RGB brightness ...")
    masks = build_object_masks(view_paths, H, W)  # (N, H, W)

    # Apply masks: zero depth outside object
    depth_masked = depth_np.copy()       # (N, H, W)
    depth_masked[~masks] = 0.0

    # 6) Unproject depth maps (object-only) to 3D point cloud
    print("Unprojecting masked depth maps to 3D (using VGGT geometry util) ...")
    # IMPORTANT: unproject_depth_map_to_point_map expects last dim = 1, so pass (N,H,W,1)
    depth_for_unproj = depth_masked[..., None]  # (N, H, W, 1)

    point_map = unproject_depth_map_to_point_map(
        depth_for_unproj, extrinsic_np, intrinsic_np
    )  # (N, H, W, 3) numpy

    N, H, W, _ = point_map.shape
    points = point_map.reshape(-1, 3)  # (N*H*W, 3)

    # Keep only finite, non-zero points
    finite = np.isfinite(points).all(axis=1)
    nonzero = ~(np.all(points == 0.0, axis=1))
    keep = finite & nonzero
    points = points[keep]

    print(f"Point cloud (object-only) has {points.shape[0]} valid points before downsampling.")

    # 6.5) Downsample to at most MAX_POINTS points
    if points.shape[0] > MAX_POINTS:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(points.shape[0], size=MAX_POINTS, replace=False)
        points = points[idx]
        print(f"Downsampled point cloud to {points.shape[0]} points.")

    # 7) Save point cloud as PLY
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    ply_path = SCENE_DIR / "point_cloud_object_only.ply"
    o3d.io.write_point_cloud(str(ply_path), pc)
    print(f"Saved object-only point cloud to {ply_path}")

    print("\nDone.")
    print(f"Scene folder (with images + cameras + 3D shape): {SCENE_DIR.resolve()}")


if __name__ == "__main__":
    main()
