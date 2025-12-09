#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import torch
import cv2
import open3d as o3d
from PIL import Image

# -----------------------------------------------------------------------------
# 1. SETUP & IMPORTS
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VGGT_REPO_ROOT = os.path.join(THIS_DIR, "vggt")
sys.path.insert(0, VGGT_REPO_ROOT)

# Import VGGT modules
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
except ImportError:
    print("Error: Could not import VGGT. Make sure the 'vggt' folder is present.")
    sys.exit(1)

# Configuration
INPUT_VIEWS_DIR = os.path.join(THIS_DIR, "views")
OUTPUT_DIR = os.path.join(THIS_DIR, "output_scene")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BG_THRESHOLD = 0.95  # Brightness threshold for mask
MAX_POINTS = 500_000

# -----------------------------------------------------------------------------
# 2. HELPERS
# -----------------------------------------------------------------------------
def normalize_depth_tensor_to_nhw(depth_tensor):
    """Converts torch depth tensor to (N, H, W) numpy array."""
    d = depth_tensor.detach().cpu().numpy()
    if d.ndim == 4 and d.shape[1] == 1:
        d = d[:, 0]
    elif d.ndim == 4 and d.shape[-1] == 1:
        d = d[..., 0]
    return d

def build_object_masks(view_paths, target_h, target_w):
    """Creates a boolean mask where True = Object, False = White Background."""
    masks = []
    for p in view_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((target_w, target_h), Image.BILINEAR)
        rgb = np.array(img).astype(np.float32) / 255.0
        gray = rgb.mean(axis=2)
        mask = gray < BG_THRESHOLD
        masks.append(mask)
    return np.stack(masks, axis=0)

def render_verification(points, colors, w2c_4x4, K, H, W, save_path):
    """
    Renders the point cloud back to the camera view to verify alignment.

    Assumes:
      - points are in WORLD coordinates (N, 3)
      - w2c_4x4 is WORLD-to-CAMERA extrinsic (4x4)
      - K is intrinsics (3x3)
    """
    # 1. Transform Points to Camera Space
    ones = np.ones((len(points), 1), dtype=np.float32)
    pts_hom = np.hstack([points.astype(np.float32), ones])  # (N,4)
    pts_cam = (w2c_4x4 @ pts_hom.T).T  # (N,4)

    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    # 2. Filter points behind camera
    valid_z = z > 0.01
    if not np.any(valid_z):
        print(f"[WARN] No valid points after z>0 filter for {save_path}")
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return

    x, y, z = x[valid_z], y[valid_z], z[valid_z]
    cur_colors = colors[valid_z]

    # 3. Project to Pixels (OpenCV-style)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = (x * fx / z) + cx
    v = (y * fy / z) + cy

    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # 4. Filter valid pixels
    valid_px = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(valid_px):
        print(f"[WARN] No valid pixels inside frame for {save_path}")
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return

    u, v, z = u[valid_px], v[valid_px], z[valid_px]
    cur_colors = cur_colors[valid_px]

    # 5. Render (Z-Buffer / Painter's Algorithm)
    # Use WHITE background so it visually matches sketches
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    # Sort by depth DESC (furthest first), so nearer points overwrite
    sort_idx = np.argsort(-z)
    uu = u[sort_idx]
    vv = v[sort_idx]
    cc = (cur_colors[sort_idx] * 255).astype(np.uint8)

    canvas[vv, uu] = cc

    # Save
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

# -----------------------------------------------------------------------------
# 3. MAIN PIPELINE
# -----------------------------------------------------------------------------
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 1. Load Images
    view_paths = sorted(
        [
            os.path.join(INPUT_VIEWS_DIR, f)
            for f in os.listdir(INPUT_VIEWS_DIR)
            if f.lower().endswith('.png')
        ]
    )
    if not view_paths:
        print("No images found in", INPUT_VIEWS_DIR)
        return
    
    print(f"Loading {len(view_paths)} images...")
    images = load_and_preprocess_images(view_paths).to(DEVICE)  # (N, 3, H, W)
    H, W = images.shape[-2:]
    print(f"Model input size: H={H}, W={W}")

    # 2. Run VGGT
    print("Running VGGT Model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE)
    model.eval()

    with torch.no_grad():
        images_batch = images.unsqueeze(0)  # (1, N, 3, H, W)
        tokens, idx = model.aggregator(images_batch)
        
        # Camera Head
        pose_enc = model.camera_head(tokens)[-1]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, (H, W))
        
        # Depth Head
        depth_tensor, _ = model.depth_head(tokens, images_batch, idx)

    # Convert to Numpy
    # IMPORTANT: from your debugging, these extrinsics behave as WORLD-TO-CAMERA (W2C).
    extrinsics_np = extrinsics.squeeze(0).cpu().numpy()  # (N, 3, 4) W2C
    intrinsics_np = intrinsics.squeeze(0).cpu().numpy()  # (N, 3, 3)
    depth_np = normalize_depth_tensor_to_nhw(depth_tensor.squeeze(0))  # (N, H, W)

    print("Extrinsics shape:", extrinsics_np.shape)
    print("Intrinsics shape:", intrinsics_np.shape)
    print("Depth shape:     ", depth_np.shape)

    # 3. Generate Point Cloud
    print("Generating Point Cloud...")
    masks = build_object_masks(view_paths, H, W)
    depth_masked = depth_np.copy()
    depth_masked[~masks] = 0.0

    depth_for_unproj = depth_masked[..., None]  # (N, H, W, 1)

    # NOTE: We pass extrinsics_np directly; unproject_depth_map_to_point_map
    # is assumed to expect WORLD-TO-CAMERA matrices here as well.
    point_map = unproject_depth_map_to_point_map(
        depth_for_unproj,
        extrinsics_np,
        intrinsics_np
    )  # (N, H, W, 3)
    
    # Flatten
    points_all = point_map.reshape(-1, 3)

    # Get Colors for PLY (per-view, resized to H,W)
    colors_list = []
    for p in view_paths:
        img = Image.open(p).convert("RGB").resize((W, H))
        colors_list.append(np.array(img).reshape(-1, 3) / 255.0)
    colors_all = np.concatenate(colors_list, axis=0)

    # Filter Valid Points
    valid_mask = np.isfinite(points_all).all(axis=1) & (np.abs(points_all).sum(axis=1) > 0)
    points_final = points_all[valid_mask]
    colors_final = colors_all[valid_mask]

    print(f"Total raw points:    {points_all.shape[0]}")
    print(f"Valid nonzero points:{points_final.shape[0]}")

    # Downsample if needed
    if len(points_final) > MAX_POINTS:
        idx_ds = np.random.choice(len(points_final), MAX_POINTS, replace=False)
        points_final = points_final[idx_ds]
        colors_final = colors_final[idx_ds]
        print(f"Downsampled to {MAX_POINTS} points")

    # Save PLY
    ply_path = os.path.join(OUTPUT_DIR, "fused_model.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_final)
    pcd.colors = o3d.utility.Vector3dVector(colors_final)
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved fused 3D point cloud: {ply_path}")

    # 4. Save Cameras & Render Verification (W2C ONLY)
    print("Saving Cameras and Verifying Alignment (W2C)...")
    
    for i, vp in enumerate(view_paths):
        name = os.path.splitext(os.path.basename(vp))[0]

        # W2C extrinsic from VGGT for this view
        w2c_3x4 = extrinsics_np[i]  # (3,4)
        K = intrinsics_np[i]        # (3,3)

        # Make full 4x4 W2C matrix
        w2c_4x4 = np.eye(4, dtype=np.float32)
        w2c_4x4[:3, :4] = w2c_3x4

        # Save JSON (name it explicitly as W2C to avoid confusion)
        cam_data = {
            "extrinsics_w2c": w2c_4x4.tolist(),
            "intrinsics": K.tolist()
        }
        json_path = os.path.join(OUTPUT_DIR, f"{name}_cam.json")
        with open(json_path, 'w') as f:
            json.dump(cam_data, f, indent=4)

        # Render verification (using W2C directly)
        render_path = os.path.join(OUTPUT_DIR, f"{name}_verification_render.png")
        render_verification(points_final, colors_final, w2c_4x4, K, H, W, render_path)

        print(f"  - View {i}:")
        print(f"      JSON:   {json_path}")
        print(f"      Render: {render_path}")

    print(f"\nSuccess! Check {OUTPUT_DIR} for:")
    print("  - fused_model.ply")
    print("  - *_cam.json (W2C extrinsics)")
    print("  - *_verification_render.png")

if __name__ == "__main__":
    main()
