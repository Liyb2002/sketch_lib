#!/usr/bin/env python3
"""
exhaustive_pose_search_fine.py

Performs a SUPER FINE 36x36x36 grid search.
- Azimuth: 36 steps (10°)
- Elevation: 36 steps (~5°)
- Roll: 36 steps (10°)

Total outputs: ~46,656 images.
"""

import os
import shutil
os.environ["OPEN3D_LOG_LEVEL"] = "error"
os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"
os.environ["FILAMENT_DISABLE_COMPUTE"] = "1"

from pathlib import Path
import math
import itertools
import numpy as np
import cv2
import open3d as o3d
from open3d.visualization import rendering
from tqdm.auto import tqdm

# ---------------- CONFIGURATION ----------------

PLY_PATH    = Path("trellis_outputs/0_trellis_gaussian.ply")
SKETCH_PATH = Path("0.png")

# Folder to dump all images
OUT_DIR = Path("pose_search_fine")
# Clean start (optional: warning, this deletes previous run)
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 160

# --- Search Grid (36 x 36 x 36) ---
# 1. Azimuth: 0 to 360 in 10 deg steps (36 steps)
AZ_VALS = np.arange(0, 360, 10)

# 2. Elevation: -85 to 85 (36 steps)
EL_VALS = np.linspace(-85, 85, 36)

# 3. Roll: 0 to 360 in 10 deg steps (36 steps)
ROLL_VALS = np.arange(0, 360, 10)


# ---------------- HELPER FUNCTIONS ----------------

def sph_dir(az_deg, el_deg):
    """Converts spherical (az, el) to cartesian direction vector."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = math.cos(el) * math.sin(az)
    y = math.cos(el) * math.cos(az)
    z = math.sin(el)
    return np.array([x, y, z], dtype=float)

def load_geometry(path: Path):
    """Loads PLY/OBJ as Mesh or PointCloud."""
    geom = o3d.io.read_point_cloud(str(path))
    if geom.is_empty():
        geom = o3d.io.read_triangle_mesh(str(path))
        if geom.is_empty():
            raise RuntimeError(f"Could not load geometry from {path}")
        geom.compute_vertex_normals()
    return geom

def make_unlit_black_material():
    """Flat black material for silhouette extraction."""
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.base_color = (0.0, 0.0, 0.0, 1.0) # Black
    mat.point_size = 2.5 
    return mat

def normalize_mask(mask: np.ndarray, size: int, margin: float = 0.1) -> np.ndarray:
    """Centers and scales the mask content."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    h_box = y_max - y_min + 1
    w_box = x_max - x_min + 1
    crop = mask[y_min:y_max+1, x_min:x_max+1]

    target_size = int((1.0 - 2 * margin) * size)
    scale = target_size / max(h_box, w_box)
    
    new_h = int(h_box * scale)
    new_w = int(w_box * scale)
    
    resized = cv2.resize(crop.astype(np.uint8) * 255, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((size, size), dtype=np.uint8)
    start_y = (size - new_h) // 2
    start_x = (size - new_w) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    return (canvas > 0).astype(np.uint8)

def load_sketch_mask(path: Path, size: int) -> np.ndarray:
    """Loads sketch image and converts to normalized binary mask."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Sketch not found: {path}")
        
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if np.mean(img) > 127: img = 255 - img
    _, mask = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    
    # Fill holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(mask)
    cv2.drawContours(sil, contours, -1, 255, thickness=-1)
    
    return normalize_mask((sil > 0).astype(np.uint8), size)

def render_solid_silhouette(renderer, center, radius, az, el) -> np.ndarray:
    """Renders 3D object as pure black mask."""
    scene = renderer.scene
    direction = sph_dir(az, el)
    eye = center + radius * direction
    up = np.array([0.0, 0.0, 1.0])
    
    cam = scene.camera
    cam.set_projection(40.0, 1.0, radius*0.1, radius*10.0, rendering.Camera.FovType.Vertical)
    cam.look_at(center, eye, up)
    
    img_o3d = renderer.render_to_image()
    img = np.asarray(img_o3d)
    
    # R channel < 200 means object
    mask = (img[:, :, 0] < 200).astype(np.uint8)
    return mask

def get_iou(mask1, mask2):
    """Intersection over Union."""
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 0.0
    return inter / union

def save_overlay(path, sketch_mask, render_mask):
    """Saves White BG + Grey Render + Red Sketch."""
    h, w = sketch_mask.shape
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    canvas[render_mask > 0] = [180, 180, 180] # Grey Render
    
    s_indices = sketch_mask > 0
    # Blend Red (0,0,255)
    existing = canvas[s_indices]
    red = np.zeros_like(existing)
    red[:] = [0, 0, 255]
    canvas[s_indices] = cv2.addWeighted(existing, 0.5, red, 0.5, 0.0)
    
    cv2.imwrite(str(path), canvas)


# ---------------- MAIN ----------------

def main():
    print("--- 📸 FINE GRID Pose Search (36x36x36) ---")
    print(f"Saving to: {OUT_DIR} (WARNING: ~46,000 files)")
    
    # 1. Load Geometry & Sketch
    geom = load_geometry(PLY_PATH)
    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    radius = 2.5 * float(np.max(bbox.get_extent()))
    
    sketch_mask = load_sketch_mask(SKETCH_PATH, IMG_SIZE)
    
    # 2. Setup Renderer
    renderer = rendering.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.add_geometry("obj", geom, make_unlit_black_material())
    
    # 3. Search Loop
    total_steps = len(AZ_VALS) * len(EL_VALS)
    
    # Rotation matrices cache for Roll (optimization)
    roll_matrices = {}
    center_pt = (IMG_SIZE//2, IMG_SIZE//2)
    for roll in ROLL_VALS:
        roll_matrices[roll] = cv2.getRotationMatrix2D(center_pt, roll, 1.0)
    
    pbar = tqdm(total=total_steps, desc="Searching")

    for az, el in itertools.product(AZ_VALS, EL_VALS):
        
        # A. Render 3D (Slow part)
        raw_mask = render_solid_silhouette(renderer, center, radius, az, el)
        norm_mask = normalize_mask(raw_mask, IMG_SIZE)
        
        # B. Apply 2D Roll (Fast part)
        for roll in ROLL_VALS:
            M = roll_matrices[roll]
            rolled_mask = cv2.warpAffine(
                norm_mask, M, (IMG_SIZE, IMG_SIZE), 
                flags=cv2.INTER_NEAREST, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
            
            score = get_iou(sketch_mask, rolled_mask)
            
            # Filename: score_az_el_roll.png
            # Adding roll to filename ensures uniqueness
            filename = f"{score:.4f}_az{int(az)}_el{int(el)}_roll{int(roll)}.png"
            save_overlay(OUT_DIR / filename, sketch_mask, rolled_mask)
            
        pbar.update(1)
            
    pbar.close()
    print("\n✅ Done! Sort folder by name to find the best IoU.")

if __name__ == "__main__":
    main()