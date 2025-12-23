import open3d as o3d
import numpy as np
import os
import json
from plyfile import PlyData

# ================= CONFIGURATION =================
INPUT_PATH = "trellis_outputs/0_trellis_gaussian.ply"
OUTPUT_DIR = "rendered_views_final"
TEMP_PCD_PATH = "temp_converted.ply"

IMG_WIDTH = 512
IMG_HEIGHT = 512
POINT_SIZE = 5.0
BG_COLOR = [1, 1, 1]

VIEWS_CONFIG = [
    (30, 30),   (90, -20),
    (150, 30),  (210, -20),
    (270, 30),  (330, -20)
]
# =================================================

def load_gaussian_splat_as_pcd(path):
    """
    Reads a Gaussian Splat PLY and converts it to a standard Open3D PointCloud.
    Extracts XYZ and Base Color (DC term of Spherical Harmonics).
    """
    print(f"Reading Gaussian Splat: {path}...")
    plydata = PlyData.read(path)
    
    vertex = plydata['vertex']
    
    # 1. Extract XYZ
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    points = np.stack([x, y, z], axis=-1)
    
    # 2. Extract Colors (Spherical Harmonics DC term)
    # The fields are usually f_dc_0, f_dc_1, f_dc_2
    # In standard GS, these are pre-sigmoid, but sometimes Trellis saves them as raw colors.
    # We will assume they act as RGB base.
    try:
        r = vertex['f_dc_0']
        g = vertex['f_dc_1']
        b = vertex['f_dc_2']
        colors = np.stack([r, g, b], axis=-1)
        
        # GS colors are often stored as SH coefficients.
        # Simple conversion: 0.28209479177387814 * color + 0.5 (approx)
        # But Trellis often exports normalized 0-1 or raw.
        # Let's try simple normalization first.
        
        # Heuristic: if max > 1.0, normalize.
        # Actually, SH 0th band is just ambient color.
        # Common formula: RGB = 0.282 * SH + 0.5
        colors = 0.28209 * colors + 0.5
        colors = np.clip(colors, 0.0, 1.0)
        
    except ValueError:
        print("Warning: No color fields found. Using Black.")
        colors = np.zeros_like(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def normalize_geometry(pcd):
    # Center
    center = pcd.get_center()
    pcd.translate(-center)
    # Scale to Unit Sphere
    max_bound = pcd.get_max_bound()
    min_bound = pcd.get_min_bound()
    extent = np.max(max_bound - min_bound)
    if extent > 0:
        pcd.scale(1.0 / extent, center=(0,0,0))
    return pcd

def get_extrinsic_matrix(azimuth, elevation, radius=1.5):
    az_rad = np.deg2rad(azimuth)
    el_rad = np.deg2rad(elevation)
    x = radius * np.cos(el_rad) * np.cos(az_rad)
    y = radius * np.cos(el_rad) * np.sin(az_rad)
    z = radius * np.sin(el_rad)
    eye = np.array([x, y, z], dtype=np.float64)
    target = np.array([0, 0, 0], dtype=np.float64)
    up = np.array([0, 0, 1], dtype=np.float64)
    z_axis = eye - target; z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    extrinsic = np.eye(4)
    extrinsic[:3, 0] = x_axis
    extrinsic[:3, 1] = y_axis
    extrinsic[:3, 2] = z_axis
    extrinsic[:3, 3] = -np.dot(np.array([x_axis, y_axis, z_axis]), eye)
    return extrinsic

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. LOAD & CONVERT
    # We use plyfile to handle the weird GS format, then pass to Open3D
    pcd = load_gaussian_splat_as_pcd(INPUT_PATH)
    
    # 2. NORMALIZE
    pcd = normalize_geometry(pcd)
    
    # 3. RENDER
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=IMG_WIDTH, height=IMG_HEIGHT, visible=False)
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray(BG_COLOR)
    opt.point_size = POINT_SIZE
    opt.light_on = True # Try toggling this if still dark

    print("Rendering...")
    for i, (az, el) in enumerate(VIEWS_CONFIG):
        # Using Radius 1.5 since object is normalized to 1.0
        extrinsic = get_extrinsic_matrix(az, el, radius=1.5)
        
        ctr = vis.get_view_control()
        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = extrinsic
        fov = 30
        focal = (IMG_HEIGHT / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(IMG_WIDTH, IMG_HEIGHT, focal, focal, IMG_WIDTH/2, IMG_HEIGHT/2)
        
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        
        save_path = os.path.join(OUTPUT_DIR, f"view_{i:02d}.png")
        vis.capture_screen_image(save_path, do_render=True)
        
        # Save JSON
        json_data = {
            "extrinsics_w2c": extrinsic.tolist(),
            "azimuth": az,
            "elevation": el,
            "fov": fov
        }
        with open(os.path.join(OUTPUT_DIR, f"view_{i:02d}.json"), 'w') as f:
            json.dump(json_data, f, indent=4)
            
        print(f"  -> Saved {save_path}")

    vis.destroy_window()
    print("Done!")

if __name__ == "__main__":
    main()