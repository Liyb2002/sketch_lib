#!/usr/bin/env python3
"""
viz_vggt_depth_and_pcloud.py

Utility to visualize VGGT outputs:

- Loads depth maps from: vggt_scene/depth_maps.npy
- Saves per-view depth visualization PNGs to: vggt_scene/depth_vis/depth_*.png
- Loads point cloud from: vggt_scene/point_cloud.ply
- Opens an interactive Open3D window to view the point cloud.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import open3d as o3d


SCENE_DIR = Path("vggt_scene")
DEPTH_NPY = SCENE_DIR / "depth_maps.npy"
POINT_CLOUD_PLY = SCENE_DIR / "point_cloud.ply"
DEPTH_VIS_DIR = SCENE_DIR / "depth_vis"


def load_depth_maps():
    """
    Load depth_maps.npy and normalize shape to (N, 1, H, W).

    Handles both:
      - (N, 1, H, W)
      - (N, H, W, 1)
    """
    if not DEPTH_NPY.exists():
        raise FileNotFoundError(f"Depth file not found: {DEPTH_NPY}")

    depth_maps = np.load(DEPTH_NPY)

    if depth_maps.ndim != 4:
        raise ValueError(f"Expected depth_maps.npy with 4 dims, got shape {depth_maps.shape}")

    # Case 1: Already (N, 1, H, W)
    if depth_maps.shape[1] == 1:
        return depth_maps

    # Case 2: (N, H, W, 1) -> transpose to (N, 1, H, W)
    if depth_maps.shape[-1] == 1:
        depth_maps = np.transpose(depth_maps, (0, 3, 1, 2))
        return depth_maps

    raise ValueError(
        f"Unsupported depth_maps shape {depth_maps.shape}, "
        f"expected (N,1,H,W) or (N,H,W,1)"
    )


def save_depth_visualizations():
    """
    Load depth_maps.npy and save one grayscale PNG per view.

    - Input: depth_maps.npy with shape (N, 1, H, W) (after normalization)
    - Output: vggt_scene/depth_vis/depth_{i}.png
    """
    DEPTH_VIS_DIR.mkdir(parents=True, exist_ok=True)

    depth_maps = load_depth_maps()  # (N, 1, H, W)
    num_views = depth_maps.shape[0]
    print(f"Loaded depth maps for {num_views} views, normalized shape: {depth_maps.shape}")

    for i in range(num_views):
        d = depth_maps[i, 0]  # (H, W)

        # Mask invalid depths (<= 0 or NaN/inf)
        valid_mask = np.isfinite(d) & (d > 0)
        if not np.any(valid_mask):
            print(f"[View {i}] No valid depth values, skipping visualization.")
            continue

        d_valid = d[valid_mask]
        d_min, d_max = d_valid.min(), d_valid.max()

        if d_max <= d_min:
            print(f"[View {i}] Depth values are nearly constant, skipping normalization.")
            d_norm = np.zeros_like(d, dtype=np.float32)
        else:
            # Normalize valid depths to [0, 1]
            d_norm = (d - d_min) / (d_max - d_min)
            d_norm[~valid_mask] = 0.0  # background black

        # Convert to 8-bit grayscale
        d_img = (d_norm * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(d_img, mode="L")
        out_path = DEPTH_VIS_DIR / f"depth_{i}.png"
        img.save(out_path)
        print(f"[View {i}] Saved depth visualization to {out_path}")


def view_point_cloud():
    """
    Load point_cloud.ply and show it in an Open3D interactive viewer.
    """
    if not POINT_CLOUD_PLY.exists():
        raise FileNotFoundError(f"Point cloud file not found: {POINT_CLOUD_PLY}")

    pcd = o3d.io.read_point_cloud(str(POINT_CLOUD_PLY))
    print(pcd)

    # Optional: downsample for speed if there are a ton of points
    # pcd = pcd.voxel_down_sample(voxel_size=0.005)

    print("Opening Open3D viewer. Close the window to end.")
    o3d.visualization.draw_geometries([pcd])


def main():
    print("=== Saving depth map visualizations ===")
    save_depth_visualizations()

    print("\n=== Viewing point cloud ===")
    view_point_cloud()


if __name__ == "__main__":
    main()
