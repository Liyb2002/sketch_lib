#!/usr/bin/env python3
"""
debug_view0_grid.py

Generates a 2x2 grid of projections to identify the coordinate mismatch.
1. Standard (As currently implemented)
2. Inverted Roll (Rotates the other way)
3. Inverted Y (Flips vertical)
4. Inverted X (Flips horizontal)
"""

import json
import math
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path

# ---------------- CONFIG ----------------
PLY_PATH     = Path("trellis_outputs/0_trellis_gaussian.ply")
CAMERAS_JSON = Path("independent_view_fits/all_cameras.json")
SEGS_DIR     = Path("segmentations/view_0") 
OUT_PATH     = Path("debug_grid.png")
IMG_SIZE     = 160

def get_camera_vectors(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = math.cos(el) * math.sin(az)
    y = math.cos(el) * math.cos(az)
    z = math.sin(el)
    pos_vec = np.array([x, y, z], dtype=np.float32)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right_vec = np.cross(pos_vec, world_up)
    if np.linalg.norm(right_vec) < 0.001: right_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right_vec /= np.linalg.norm(right_vec)
    up_vec = np.cross(right_vec, pos_vec)
    up_vec /= np.linalg.norm(up_vec)
    return pos_vec, up_vec, right_vec

def project_raw(points, center, radius, az, el, img_size):
    pos_vec, up_vec, right_vec = get_camera_vectors(az, el)
    eye = center + pos_vec * radius
    vec = points - eye
    view_dir = -pos_vec 
    
    dist_z = np.dot(vec, view_dir)
    dist_y = np.dot(vec, up_vec)
    dist_x = np.dot(vec, right_vec)
    
    fov_rad = math.radians(40)
    focal_length = (img_size / 2) / math.tan(fov_rad / 2)
    safe_z = np.maximum(dist_z, 0.001)
    
    # Raw normalized coordinates (0 center)
    u_raw = (focal_length * dist_x / safe_z)
    v_raw = -(focal_length * dist_y / safe_z) # Standard GL: Y is up, Image Y is down
    
    return u_raw, v_raw, dist_z

def rotate_points(u, v, roll_deg, img_size):
    cx, cy = 0, 0 # We rotate around center (0,0) before shifting
    theta = math.radians(roll_deg) # Try positive first
    
    # Standard Rotation Matrix
    # x' = x cos - y sin
    # y' = x sin + y cos
    new_u = u * math.cos(theta) - v * math.sin(theta)
    new_v = u * math.sin(theta) + v * math.cos(theta)
    
    # Shift to Image Coordinates (0,0 is top-left)
    final_u = new_u + (img_size / 2)
    final_v = new_v + (img_size / 2)
    
    return np.stack([final_u, final_v], axis=1).astype(int)

def draw_plot(points, center, radius, az, el, roll, img_size, mode, mask_composite):
    u_raw, v_raw, dists = project_raw(points, center, radius, az, el, img_size)
    
    # Apply Mode Modifications
    if mode == "Standard":
        # Search script used cv2.warpAffine with POSITIVE angle (CCW).
        # To match points to that, we should likely rotate points CCW.
        # My previous code used -roll (CW). Let's try roll (CCW).
        uvs = rotate_points(u_raw, v_raw, -roll, img_size) # Previous attempt (CW)
        
    elif mode == "Inverted Roll":
        uvs = rotate_points(u_raw, v_raw, roll, img_size) # CCW
        
    elif mode == "Inverted Y":
        # Flip V before rotation
        uvs = rotate_points(u_raw, -v_raw, -roll, img_size)
        
    elif mode == "Inverted X":
        # Flip U before rotation
        uvs = rotate_points(-u_raw, v_raw, -roll, img_size)

    # Filter
    valid = (dists > 0) & (uvs[:,0]>=0) & (uvs[:,0]<img_size) & (uvs[:,1]>=0) & (uvs[:,1]<img_size)
    final_uvs = uvs[valid]
    
    # Draw
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for u, v in final_uvs:
        canvas[v, u] = [200, 200, 200]
        
    # Overlay Red Mask
    red_layer = np.zeros_like(canvas)
    red_layer[:, :, 2] = mask_composite
    mask_indices = mask_composite > 50
    canvas[mask_indices] = cv2.addWeighted(canvas[mask_indices], 0.5, red_layer[mask_indices], 0.5, 0.0)
    
    # Add Label
    cv2.putText(canvas, mode, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas

def main():
    with open(CAMERAS_JSON, 'r') as f: params = json.load(f)["view_0"]
    az, el, roll = params['azimuth'], params['elevation'], params['roll']
    
    pcd = o3d.io.read_point_cloud(str(PLY_PATH))
    if len(pcd.points) == 0:
        mesh = o3d.io.read_triangle_mesh(str(PLY_PATH))
        pcd = mesh.sample_points_poisson_disk(5000)
    points = np.asarray(pcd.points)
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    radius = 2.5 * float(np.max(bbox.get_extent()))
    if radius <= 0: radius = 1.0

    # Load Mask
    mask_composite = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    if SEGS_DIR.exists():
        for p in SEGS_DIR.glob("*_mask.png"):
            m = cv2.imread(str(p), 0)
            if m is not None:
                mask_composite = np.maximum(mask_composite, cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=0))

    # Generate 4 Views
    img1 = draw_plot(points, center, radius, az, el, roll, IMG_SIZE, "Standard", mask_composite)
    img2 = draw_plot(points, center, radius, az, el, roll, IMG_SIZE, "Inverted Roll", mask_composite)
    img3 = draw_plot(points, center, radius, az, el, roll, IMG_SIZE, "Inverted Y", mask_composite)
    img4 = draw_plot(points, center, radius, az, el, roll, IMG_SIZE, "Inverted X", mask_composite)
    
    # Combine into 2x2 Grid
    top = np.hstack([img1, img2])
    bot = np.hstack([img3, img4])
    grid = np.vstack([top, bot])
    
    cv2.imwrite(str(OUT_PATH), grid)
    print(f"✅ Saved debug grid to {OUT_PATH}")
    print("Look at the image. Which one aligns correctly?")
    print("1. Top-Left: Standard (-roll)")
    print("2. Top-Right: Inverted Roll (+roll)")
    print("3. Bottom-Left: Inverted Y")
    print("4. Bottom-Right: Inverted X")

if __name__ == "__main__":
    main()