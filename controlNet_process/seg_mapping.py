import os
import json
import numpy as np
import cv2
import open3d as o3d
import glob
import re

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")

# Input Folders
# 1. Previous output (3D model + cameras)
SCENE_DIR = os.path.join(SKETCH_ROOT, "3d_reconstruction")

# 2. 2D Segmentations (User specified: /sketch/segmentation)
# Inside here should be folders like 'view_0', 'view_1', etc.
SEG_DIR = os.path.join(SKETCH_ROOT, "segmentation")

# 3. Output
OUTPUT_DIR = os.path.join(SKETCH_ROOT, "final_overlays")
PLY_PATH = os.path.join(SCENE_DIR, "fused_model.ply")

# 1. Mask Expansion (Pixels) - How lenient we are with the 2D mask edges
TOLERANCE = 1

# 2. Occlusion Threshold (Depth Units)
OCCLUSION_THRESHOLD = 0.05

COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
]

# -----------------------------------------------------------------------------
# MATH HELPERS
# -----------------------------------------------------------------------------
def get_scaled_intrinsics(K_orig, src_w, src_h, target_w, target_h):
    scale_x = target_w / src_w
    scale_y = target_h / src_h
    K_new = K_orig.copy()
    K_new[0, 0] *= scale_x; K_new[0, 2] *= scale_x
    K_new[1, 1] *= scale_y; K_new[1, 2] *= scale_y
    return K_new

def project_points(points, w2c_4x4, K, H, W):
    ones = np.ones((len(points), 1))
    pts_hom = np.hstack([points, ones])
    pts_cam = (w2c_4x4 @ pts_hom.T).T

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

def render_painter_algo(u, v, z, valid_mask, colors, H, W):
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = z[valid_mask]
    c_valid = colors[valid_mask]

    sort_idx = np.argsort(-z_valid)
    uu = u_valid[sort_idx]
    vv = v_valid[sort_idx]
    cc = (c_valid[sort_idx] * 255).astype(np.uint8)

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    cc_bgr = cc[:, [2, 1, 0]]
    canvas[vv, uu] = cc_bgr
    return canvas

def compute_depth_buffer(u, v, z, valid_mask, H, W):
    """
    Creates a Z-Buffer (Depth Map) of the scene.
    Stores the MINIMUM depth at every pixel.
    """
    # Initialize with Infinity
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)
    
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = z[valid_mask]
    
    # Efficiently compute min depth per pixel using numpy
    np.minimum.at(depth_buffer, (v_valid, u_valid), z_valid)
    
    return depth_buffer

def save_labeled_ply(points, u, v, z, valid_mask, mask_paths, view_id, H, W):
    """
    Saves FULL cloud. Checks Mask AND Depth Buffer to handle occlusion.
    """
    cloud_colors = np.zeros((len(points), 3), dtype=np.float64)

    # 1. Compute Surface Depth (Z-Buffer)
    depth_buffer = compute_depth_buffer(u, v, z, valid_mask, H, W)

    # Indices of points that land on screen
    valid_indices = np.where(valid_mask)[0]
    
    # Extract data for valid points
    u_val = u[valid_indices]
    v_val = v[valid_indices]
    z_val = z[valid_indices]

    # 2. Compute Visibility Mask (Occlusion Check)
    # Get the "surface depth" at the pixel each point projects to
    surface_z = depth_buffer[v_val, u_val]
    
    # A point is visible if its depth is close to the surface depth
    # z <= surface + threshold
    is_visible = z_val <= (surface_z + OCCLUSION_THRESHOLD)

    # Filter our working set to ONLY visible points
    visible_indices = valid_indices[is_visible]
    u_vis = u_val[is_visible]
    v_vis = v_val[is_visible]
    
    # 3. Iterate Masks and Paint (Only Visible Points)
    for i, mpath in enumerate(mask_paths):
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        
        # Dilate mask (Tolerance)
        if TOLERANCE > 0:
            kernel = np.ones((TOLERANCE*2+1, TOLERANCE*2+1), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        b, g, r = COLORS[i % len(COLORS)]
        rgb_color = np.array([r, g, b]) / 255.0

        # Check mask only at visible pixels
        mask_vals = mask[v_vis, u_vis]
        mask_hits = mask_vals > 0
        
        if np.any(mask_hits):
            # Update colors of original points
            final_hit_indices = visible_indices[mask_hits]
            cloud_colors[final_hit_indices] = rgb_color

    # 4. Save
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    pcd_out.colors = o3d.utility.Vector3dVector(cloud_colors)
    
    save_path = os.path.join(OUTPUT_DIR, f"{view_id}_labeled.ply")
    o3d.io.write_point_cloud(save_path, pcd_out)

def overlay_masks(base_img, mask_paths):
    overlay = base_img.copy()
    alpha = 0.5
    for i, mpath in enumerate(mask_paths):
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        if mask.shape != base_img.shape[:2]:
            mask = cv2.resize(mask, (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = COLORS[i % len(COLORS)]
        colored_mask = np.zeros_like(base_img)
        colored_mask[:] = color
        binary_mask = mask > 0
        overlay[binary_mask] = cv2.addWeighted(
            overlay[binary_mask], 1 - alpha, colored_mask[binary_mask], alpha, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)
    return overlay

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(PLY_PATH):
        print(f"Error: Could not find fused model at {PLY_PATH}")
        return

    print(f"Loading 3D model: {PLY_PATH}")
    pcd = o3d.io.read_point_cloud(PLY_PATH)
    points = np.asarray(pcd.points)
    orig_colors = np.asarray(pcd.colors) 

    # Look for cameras in the 3d_reconstruction folder
    cam_files = glob.glob(os.path.join(SCENE_DIR, "*_cam.json"))
    if not cam_files: 
        print(f"Error: No camera JSON files found in {SCENE_DIR}")
        return

    for cam_json_path in cam_files:
        filename = os.path.basename(cam_json_path)
        match = re.search(r"(view_\d+)_cam\.json", filename)
        if not match: continue
        view_id = match.group(1)

        # Look for view folder inside segmentation directory
        seg_folder = os.path.join(SEG_DIR, view_id)
        if not os.path.exists(seg_folder): 
            print(f"Warning: No segmentation folder found for {view_id} at {seg_folder}")
            continue
            
        mask_paths = glob.glob(os.path.join(seg_folder, "*_mask.png"))
        if not mask_paths: 
            print(f"Warning: No masks found in {seg_folder}")
            continue
        
        print(f"Processing {view_id}...")

        temp_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
        target_h, target_w = temp_mask.shape

        verif_img_path = os.path.join(SCENE_DIR, f"{view_id}_verification_render.png")
        source_h, source_w = 0, 0
        if os.path.exists(verif_img_path):
            v_img = cv2.imread(verif_img_path)
            if v_img is not None:
                source_h, source_w = v_img.shape[:2]

        with open(cam_json_path, 'r') as f:
            cam_data = json.load(f)
        w2c = np.array(cam_data["extrinsics_w2c"])
        K_orig = np.array(cam_data["intrinsics"])

        if source_w == 0:
            source_w = K_orig[0, 2] * 2
            source_h = K_orig[1, 2] * 2

        K_scaled = get_scaled_intrinsics(K_orig, source_w, source_h, target_w, target_h)

        # 1. Project
        u, v, z, valid_mask = project_points(points, w2c, K_scaled, target_h, target_w)

        # 2. Render 2D Overlay
        render_img = render_painter_algo(u, v, z, valid_mask, orig_colors, target_h, target_w)
        final_comp = overlay_masks(render_img, mask_paths)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{view_id}_overlay.png"), final_comp)

        # 3. Save Labeled PLY (With Occlusion Check)
        save_labeled_ply(points, u, v, z, valid_mask, mask_paths, view_id, target_h, target_w)

if __name__ == "__main__":
    main()