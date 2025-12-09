import os
import json
import numpy as np
import cv2

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VIEWS_DIR = os.path.join(THIS_DIR, "views")
INPUT_DIR = os.path.join(THIS_DIR, "outputs")
RENDER_DIR = os.path.join(THIS_DIR, "renders")

def save_ply(points, colors, filename):
    print(f"Saving {filename} ({len(points)} points)...")
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def get_local_points(depth_map, rgb_img, K):
    """Generates 3D points in the Camera's local space (OpenCV convention)."""
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u, v = u.flatten(), v.flatten()
    depth = depth_map.flatten()
    
    valid_mask = ~np.isnan(depth) & (depth > 0)
    u, v, depth = u[valid_mask], v[valid_mask], depth[valid_mask]
    colors = rgb_img.reshape(-1, 3)[valid_mask]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    # OpenCV: +X Right, +Y Down, +Z Forward
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth
    
    return np.stack([x_cam, y_cam, z_cam], axis=1), colors

def apply_transform(points, matrix):
    """Applies 4x4 matrix to Nx3 points."""
    ones = np.ones((len(points), 1))
    pts_hom = np.hstack([points, ones])
    return (matrix @ pts_hom.T).T[:, :3]

def main():
    if not os.path.exists(RENDER_DIR): os.makedirs(RENDER_DIR)

    image_files = sorted([f for f in os.listdir(VIEWS_DIR) if f.lower().endswith(('.png', '.jpg'))])
    
    # We will collect points for 3 different hypothesis modes
    points_A, colors_A = [], [] # Mode A: Standard Inverse (W2C -> C2W)
    points_B, colors_B = [], [] # Mode B: OpenGL Conversion (Flip Y/Z)
    points_C, colors_C = [], [] # Mode C: Direct Multiply (C2W assumption)

    print("Processing views...")
    for f in image_files:
        name = os.path.splitext(f)[0]
        depth_path = os.path.join(INPUT_DIR, f"{name}_depth.npy")
        cam_path = os.path.join(INPUT_DIR, f"{name}_cam.json")
        
        if not (os.path.exists(depth_path) and os.path.exists(cam_path)): continue
        
        # Load Data
        img = cv2.cvtColor(cv2.imread(os.path.join(VIEWS_DIR, f)), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)
        
        with open(cam_path) as json_file:
            cam_data = json.load(json_file)
        
        extrinsic = np.array(cam_data['extrinsics']) # 4x4 or 3x4
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0,0,0,1]])
            
        K = np.array(cam_data['intrinsics'])
        
        # 1. Get Local Points (OpenCV Style: +Z forward, +Y down)
        local_pts, cols = get_local_points(depth, img, K)
        
        # --- MODE A: Standard W2C Inverse ---
        # Assumes extrinsic is World-to-Camera. We invert to get Camera-to-World.
        c2w_A = np.linalg.inv(extrinsic)
        pts_A = apply_transform(local_pts, c2w_A)
        points_A.append(pts_A); colors_A.append(cols)
        
        # --- MODE B: OpenGL Fix (Flip Y and Z) ---
        # Many research models expect cameras to look down -Z with +Y up.
        # We transform our OpenCV points to that system before applying matrix.
        local_pts_gl = local_pts.copy()
        local_pts_gl[:, 1] *= -1 # Flip Y
        local_pts_gl[:, 2] *= -1 # Flip Z
        
        c2w_B = np.linalg.inv(extrinsic)
        pts_B = apply_transform(local_pts_gl, c2w_B)
        points_B.append(pts_B); colors_B.append(cols)

        # --- MODE C: Direct C2W ---
        # Assumes extrinsic is ALREADY Camera-to-World.
        pts_C = apply_transform(local_pts, extrinsic)
        points_C.append(pts_C); colors_C.append(cols)

    # Save Results
    print("Saving PLY files...")
    save_ply(np.vstack(points_A), np.vstack(colors_A), os.path.join(RENDER_DIR, "fused_mode_A.ply"))
    save_ply(np.vstack(points_B), np.vstack(colors_B), os.path.join(RENDER_DIR, "fused_mode_B.ply"))
    save_ply(np.vstack(points_C), np.vstack(colors_C), os.path.join(RENDER_DIR, "fused_mode_C.ply"))
    
    print("\nDONE!")
    print("1. Open MeshLab.")
    print("2. Import 'fused_mode_A.ply', 'fused_mode_B.ply', and 'fused_mode_C.ply'.")
    print("3. One of them will look like a single solid object. The others will be scattered.")

if __name__ == "__main__":
    main()