import os
import sys
import shutil
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

def normalize_depth_tensor_to_nhw(depth_tensor):
    """Converts torch depth tensor to (N, H, W) numpy array."""
    d = depth_tensor.detach().cpu().numpy()
    if d.ndim == 4 and d.shape[1] == 1: d = d[:, 0]
    elif d.ndim == 4 and d.shape[-1] == 1: d = d[..., 0]
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

def render_verification(points, colors, c2w, K, H, W, save_path):
    """
    Renders the point cloud back to the camera view to verify alignment.
    """
    # 1. Invert C2W to get World-to-Camera (W2C)
    # Because we are moving from the Global Point Cloud -> Camera
    w2c = np.linalg.inv(c2w)

    # 2. Transform Points to Camera Space
    # P_cam = W2C @ P_world
    ones = np.ones((len(points), 1))
    pts_hom = np.hstack([points, ones])
    pts_cam = (w2c @ pts_hom.T).T

    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    # 3. Filter points behind camera
    valid_z = z > 0.01
    x, y, z = x[valid_z], y[valid_z], z[valid_z]
    cur_colors = colors[valid_z]

    # 4. Project to Pixels
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = (x * fx / z) + cx
    v = (y * fy / z) + cy

    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # 5. Filter valid pixels
    valid_px = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[valid_px], v[valid_px], z[valid_px]
    cur_colors = cur_colors[valid_px]

    # 6. Render (Z-Buffer / Painter's Algorithm)
    canvas = np.zeros((H, W, 3), dtype=np.uint8) # Black background
    
    # Sort by depth descending (draw furthest first)
    sort_idx = np.argsort(-z)
    canvas[v[sort_idx], u[sort_idx]] = (cur_colors[sort_idx] * 255).astype(np.uint8)

    # Save
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 1. Load Images
    view_paths = sorted([os.path.join(INPUT_VIEWS_DIR, f) for f in os.listdir(INPUT_VIEWS_DIR) if f.endswith('.png')])
    if not view_paths:
        print("No images found.")
        return
    
    print(f"Loading {len(view_paths)} images...")
    images = load_and_preprocess_images(view_paths).to(DEVICE) # (N, 3, H, W)
    H, W = images.shape[-2:]

    # 2. Run VGGT
    print("Running VGGT Model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE)
    model.eval()

    with torch.no_grad():
        images_batch = images.unsqueeze(0) # (1, N, 3, H, W)
        tokens, idx = model.aggregator(images_batch)
        
        # Camera Head
        pose_enc = model.camera_head(tokens)[-1]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, (H, W))
        
        # Depth Head
        depth_tensor, _ = model.depth_head(tokens, images_batch, idx)

    # Convert to Numpy
    # These extrinsics are C2W (Camera-to-World) because they work with the unproject function
    extrinsics_np = extrinsics.squeeze(0).cpu().numpy() # (N, 3, 4)
    intrinsics_np = intrinsics.squeeze(0).cpu().numpy() # (N, 3, 3)
    depth_np = normalize_depth_tensor_to_nhw(depth_tensor.squeeze(0)) # (N, H, W)

    # 3. Generate Point Cloud (The method you verified)
    print("Generating Point Cloud...")
    masks = build_object_masks(view_paths, H, W)
    depth_masked = depth_np.copy()
    depth_masked[~masks] = 0.0

    # Pass to VGGT utility
    depth_for_unproj = depth_masked[..., None] # (N, H, W, 1)
    point_map = unproject_depth_map_to_point_map(depth_for_unproj, extrinsics_np, intrinsics_np)
    
    # Flatten
    points_all = point_map.reshape(-1, 3)
    
    # Get Colors for PLY
    # Resizing original images to match model H,W
    colors_list = []
    for p in view_paths:
        img = Image.open(p).convert("RGB").resize((W, H))
        colors_list.append(np.array(img).reshape(-1, 3) / 255.0)
    colors_all = np.concatenate(colors_list, axis=0)

    # Filter Valid Points
    valid_mask = np.isfinite(points_all).all(axis=1) & (np.abs(points_all).sum(axis=1) > 0)
    points_final = points_all[valid_mask]
    colors_final = colors_all[valid_mask]

    # Downsample if needed
    if len(points_final) > MAX_POINTS:
        idx = np.random.choice(len(points_final), MAX_POINTS, replace=False)
        points_final = points_final[idx]
        colors_final = colors_final[idx]

    # Save PLY
    ply_path = os.path.join(OUTPUT_DIR, "fused_model.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_final)
    pcd.colors = o3d.utility.Vector3dVector(colors_final)
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved Correct 3D Shape: {ply_path}")

    # 4. Save Cameras & Render Verification
    print("Saving Cameras and Verifying Alignment...")
    
    for i in range(len(view_paths)):
        name = os.path.basename(view_paths[i]).split('.')[0]
        
        # Save JSON
        c2w = extrinsics_np[i] # 3x4
        K = intrinsics_np[i]   # 3x3
        
        # Make square 4x4 for JSON convention
        c2w_4x4 = np.vstack([c2w, [0,0,0,1]])
        
        cam_data = {
            "extrinsics_c2w": c2w_4x4.tolist(),
            "intrinsics": K.tolist()
        }
        with open(os.path.join(OUTPUT_DIR, f"{name}_cam.json"), 'w') as f:
            json.dump(cam_data, f, indent=4)

        # RENDER CHECK
        # We take the fused point cloud and project it BACK into this camera.
        # If the camera is correct, the image will look like the input.
        render_path = os.path.join(OUTPUT_DIR, f"{name}_verification_render.png")
        render_verification(points_final, colors_final, c2w_4x4, K, H, W, render_path)
        print(f"  - View {i}: Saved JSON and Render -> {render_path}")

    print(f"\nSuccess! Check {OUTPUT_DIR} for the PLY, Camera JSONs, and Verification Renders.")

if __name__ == "__main__":
    main()