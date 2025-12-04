#!/usr/bin/env python3
"""
debug_view0_projection_v2.py

Fixes:
1. Reverses projection direction (look AT object, not away).
2. Forces image save even if points are missing.
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
OUT_PATH     = Path("debug_view0_projection.png")
IMG_SIZE     = 160

def get_camera_vectors(az_deg, el_deg):
    """
    Returns the camera basis vectors.
    pos_vec: Vector from Center -> Camera (The position on the sphere)
    up_vec:  Vector pointing 'Up' relative to the camera
    right_vec: Vector pointing 'Right' relative to the camera
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    
    # Spherical to Cartesian (Position on Unit Sphere)
    x = math.cos(el) * math.sin(az)
    y = math.cos(el) * math.cos(az)
    z = math.sin(el)
    pos_vec = np.array([x, y, z], dtype=np.float32)
    
    # World Up (Z-up)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    
    # Right Vector (Cross product of Position and World Up)
    # Note: Position vector points OUT from center.
    right_vec = np.cross(pos_vec, world_up)
    if np.linalg.norm(right_vec) < 0.001: 
        right_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right_vec /= np.linalg.norm(right_vec)
    
    # Camera Up (Cross product of Right and Position)
    up_vec = np.cross(right_vec, pos_vec)
    up_vec /= np.linalg.norm(up_vec)
    
    return pos_vec, up_vec, right_vec

def project_points_manual(points, center, radius, az, el, img_size):
    pos_vec, up_vec, right_vec = get_camera_vectors(az, el)
    eye = center + pos_vec * radius
    
    # Vector from Eye to Point
    vec = points - eye
    
    # --- FIX IS HERE ---
    # The camera looks AT the center. 
    # So the View Direction is -pos_vec (Center - Eye).
    view_dir = -pos_vec 
    
    # Project vectors onto camera basis
    # Z (Depth) = Dot product with View Direction
    dist_z = np.dot(vec, view_dir)
    
    # Y (Vertical) = Dot product with Up Vector
    dist_y = np.dot(vec, up_vec)
    
    # X (Horizontal) = Dot product with Right Vector
    dist_x = np.dot(vec, right_vec)
    
    # Perspective Projection
    fov_rad = math.radians(40)
    focal_length = (img_size / 2) / math.tan(fov_rad / 2)
    
    # Avoid div by zero
    safe_z = np.maximum(dist_z, 0.001)
    
    # u = f * (x / z) + cx
    u = (focal_length * dist_x / safe_z) + (img_size / 2)
    
    # v = -f * (y / z) + cy  (Negative because Image Y is down, World Y is up)
    v = -(focal_length * dist_y / safe_z) + (img_size / 2)
    
    uv = np.stack([u, v], axis=1).astype(int)
    return uv, dist_z

def main():
    print("--- 🔬 Debugging View 0 Projection (v2) ---")

    # 1. Load Data
    if not CAMERAS_JSON.exists(): 
        print("❌ all_cameras.json not found")
        return
        
    with open(CAMERAS_JSON, 'r') as f: cameras = json.load(f)
    
    if "view_0" not in cameras:
        print("❌ view_0 not in json")
        return

    params = cameras["view_0"]
    az, el, roll = params['azimuth'], params['elevation'], params['roll']
    print(f"Cam: Az {az}, El {el}, Roll {roll}")

    pcd = o3d.io.read_point_cloud(str(PLY_PATH))
    if len(pcd.points) == 0:
        mesh = o3d.io.read_triangle_mesh(str(PLY_PATH))
        pcd = mesh.sample_points_poisson_disk(5000)
    points = np.asarray(pcd.points)
    
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    radius = 2.5 * float(np.max(bbox.get_extent()))
    if radius <= 0: radius = 1.0

    # 2. Project
    uvs, dists = project_points_manual(points, center, radius, az, el, IMG_SIZE)
    
    # 3. Stats
    print(f"\nDepth Stats (dist_z):")
    print(f"   Min: {np.min(dists):.4f}")
    print(f"   Max: {np.max(dists):.4f}")
    
    valid_mask = (dists > 0)
    print(f"   Points with positive depth: {np.sum(valid_mask)} / {len(points)}")
    
    # 4. Draw Image
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    # Filter points inside image
    on_screen = valid_mask & \
                (uvs[:, 0] >= 0) & (uvs[:, 0] < IMG_SIZE) & \
                (uvs[:, 1] >= 0) & (uvs[:, 1] < IMG_SIZE)
                
    final_uvs = uvs[on_screen]
    print(f"   Points on screen: {len(final_uvs)}")

    # Apply Roll Correction (Visualization Only)
    if roll != 0 and len(final_uvs) > 0:
        print(f"   Applying roll {roll}...")
        cx, cy = IMG_SIZE/2, IMG_SIZE/2
        theta = math.radians(-roll) 
        
        px = final_uvs[:, 0] - cx
        py = final_uvs[:, 1] - cy
        
        new_px = px * math.cos(theta) - py * math.sin(theta)
        new_py = px * math.sin(theta) + py * math.cos(theta)
        
        final_uvs[:, 0] = (new_px + cx).astype(int)
        final_uvs[:, 1] = (new_py + cy).astype(int)
        
        # Clip again after rotation
        mask_roll = (final_uvs[:, 0] >= 0) & (final_uvs[:, 0] < IMG_SIZE) & \
                    (final_uvs[:, 1] >= 0) & (final_uvs[:, 1] < IMG_SIZE)
        final_uvs = final_uvs[mask_roll]

    # Draw Points (Grey)
    for u, v in final_uvs:
        canvas[v, u] = [200, 200, 200]

    # Overlay Masks (Red)
    mask_composite = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    if SEGS_DIR.exists():
        for p in SEGS_DIR.glob("*_mask.png"):
            m = cv2.imread(str(p), 0)
            if m is not None:
                m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                mask_composite = np.maximum(mask_composite, m)
    
    red_layer = np.zeros_like(canvas)
    red_layer[:, :, 2] = mask_composite
    
    mask_indices = mask_composite > 50
    canvas[mask_indices] = cv2.addWeighted(canvas[mask_indices], 0.5, red_layer[mask_indices], 0.5, 0.0)

    # Force Save
    cv2.imwrite(str(OUT_PATH), canvas)
    print(f"\n✅ Saved debug image to {OUT_PATH}")

if __name__ == "__main__":
    main()