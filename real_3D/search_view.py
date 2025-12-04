#!/usr/bin/env python3
"""
independent_view_search.py

Iterates through all 6 views (view_0.png ... view_5.png).
Performs an independent Coarse + Fine pose search for EACH view.
Saves overlays, renders, and a single JSON file with all camera parameters.
"""

import os
import shutil
import json
import math
import itertools
import numpy as np
import cv2
import open3d as o3d
from open3d.visualization import rendering
from tqdm.auto import tqdm
from pathlib import Path

# ---------------- CONFIGURATION ----------------

PLY_PATH    = Path("trellis_outputs/0_trellis_gaussian.ply")
VIEWS_DIR   = Path("views")        # Where input sketches are
OUT_DIR     = Path("independent_view_fits") # Output folder

IMG_SIZE    = 160
TOP_K_CANDIDATES = 1 # Only refine the single best coarse match for speed

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
    geom = o3d.io.read_point_cloud(str(path))
    if geom.is_empty():
        geom = o3d.io.read_triangle_mesh(str(path))
        if geom.is_empty():
            raise RuntimeError(f"Could not load geometry from {path}")
        geom.compute_vertex_normals()
    return geom

def normalize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Centers and scales the mask content."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    h_box = y_max - y_min + 1
    w_box = x_max - x_min + 1
    crop = mask[y_min:y_max+1, x_min:x_max+1]

    # Target 85% fill
    target_size = int(size * 0.85)
    scale = target_size / max(h_box, w_box)
    
    new_h = int(h_box * scale)
    new_w = int(w_box * scale)
    
    resized = cv2.resize(crop.astype(np.uint8) * 255, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((size, size), dtype=np.uint8)
    start_y = (size - new_h) // 2
    start_x = (size - new_w) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    return (canvas > 0).astype(np.uint8)

def load_sketch_mask(view_idx: int, size: int) -> np.ndarray:
    """Tries to find view_X.png or X.png."""
    candidates = [
        VIEWS_DIR / f"view_{view_idx}.png",
        VIEWS_DIR / f"{view_idx}.png"
    ]
    path = None
    for p in candidates:
        if p.exists():
            path = p
            break
            
    if path is None:
        return None
        
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
        
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
    # Fix clipping planes
    cam.set_projection(40.0, 1.0, radius*0.01, radius*100.0, rendering.Camera.FovType.Vertical)
    cam.look_at(center, eye, up)
    
    img_o3d = renderer.render_to_image()
    img = np.asarray(img_o3d)
    
    # R channel < 200 means object
    mask = (img[:, :, 0] < 200).astype(np.uint8)
    return mask

def get_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 0.0
    return inter / union

def save_overlay(path, sketch_mask, render_mask):
    h, w = sketch_mask.shape
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    canvas[render_mask > 0] = [180, 180, 180] 
    
    s_indices = sketch_mask > 0
    existing = canvas[s_indices]
    red = np.zeros_like(existing)
    red[:] = [0, 0, 255]
    canvas[s_indices] = cv2.addWeighted(existing, 0.5, red, 0.5, 0.0)
    
    cv2.imwrite(str(path), canvas)


# ---------------- SEARCH LOGIC ----------------

def find_best_pose(renderer, center, radius, sketch_mask):
    """Runs Coarse (20deg) + Fine (1deg) search for ONE view."""
    
    # --- STAGE 1: COARSE ---
    AZ_COARSE = np.arange(0, 360, 20)
    EL_COARSE = np.arange(-80, 81, 20)
    ROLL_COARSE = np.arange(0, 360, 20)
    
    # Roll Cache
    roll_matrices = {}
    center_pt = (IMG_SIZE//2, IMG_SIZE//2)
    for r in ROLL_COARSE:
        roll_matrices[r] = cv2.getRotationMatrix2D(center_pt, r, 1.0)
        
    best_coarse_score = -1.0
    best_coarse_params = (0, 0, 0)
    
    for az, el in itertools.product(AZ_COARSE, EL_COARSE):
        raw_mask = render_solid_silhouette(renderer, center, radius, az, el)
        norm_mask = normalize_mask(raw_mask, IMG_SIZE)
        
        for roll in ROLL_COARSE:
            M = roll_matrices[roll]
            rolled_mask = cv2.warpAffine(
                norm_mask, M, (IMG_SIZE, IMG_SIZE), 
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            score = get_iou(sketch_mask, rolled_mask)
            if score > best_coarse_score:
                best_coarse_score = score
                best_coarse_params = (az, el, roll)
                
    # --- STAGE 2: FINE ---
    c_az, c_el, c_roll = best_coarse_params
    
    # Range +/- 10 degrees
    az_range = range(int(c_az - 10), int(c_az + 11))
    el_range = range(int(c_el - 10), int(c_el + 11))
    roll_range = range(int(c_roll - 10), int(c_roll + 11))
    
    best_fine_score = -1.0
    best_fine_params = best_coarse_params
    best_fine_mask = None
    
    for f_az in az_range:
        for f_el in el_range:
            render_az = f_az % 360
            render_el = max(-89, min(89, f_el))
            
            # Render 3D
            raw_mask = render_solid_silhouette(renderer, center, radius, render_az, render_el)
            norm_mask = normalize_mask(raw_mask, IMG_SIZE)
            
            # Check rolls
            for f_roll in roll_range:
                render_roll = f_roll % 360
                M = cv2.getRotationMatrix2D(center_pt, render_roll, 1.0)
                rolled_mask = cv2.warpAffine(
                    norm_mask, M, (IMG_SIZE, IMG_SIZE), 
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                score = get_iou(sketch_mask, rolled_mask)
                
                if score > best_fine_score:
                    best_fine_score = score
                    best_fine_params = (render_az, render_el, render_roll)
                    best_fine_mask = rolled_mask.copy()
                    
    return best_fine_score, best_fine_params, best_fine_mask

# ---------------- MAIN ----------------

def main():
    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"--- 🔍 Independent View Search (Coarse+Fine) ---")

    # 1. Setup Renderer & Geometry
    geom = load_geometry(PLY_PATH)
    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    radius = 2.5 * float(np.max(bbox.get_extent()))
    if radius <= 0: radius = 1.0
    
    renderer = rendering.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.base_color = (0.0, 0.0, 0.0, 1.0)
    mat.point_size = 2.5 
    renderer.scene.add_geometry("obj", geom, mat)
    
    all_cameras = {}
    
    # 2. Iterate Views 0 to 5
    pbar = tqdm(total=6, desc="Fitting Views")
    
    for i in range(6):
        sketch_mask = load_sketch_mask(i, IMG_SIZE)
        
        if sketch_mask is None:
            print(f"⚠️ Skipping View {i}: File not found.")
            all_cameras[f"view_{i}"] = {"error": "not_found"}
            pbar.update(1)
            continue
            
        # Run Search
        score, (az, el, roll), best_mask = find_best_pose(renderer, center, radius, sketch_mask)
        
        # Save Images
        save_overlay(OUT_DIR / f"view_{i}_overlay.png", sketch_mask, best_mask)
        cv2.imwrite(str(OUT_DIR / f"view_{i}_render.png"), best_mask * 255)
        
        # Store Data
        all_cameras[f"view_{i}"] = {
            "iou_score": float(score),
            "azimuth": int(az),
            "elevation": int(el),
            "roll": int(roll)
        }
        
        pbar.update(1)
        
    pbar.close()
    
    # 3. Save Master JSON
    json_path = OUT_DIR / "all_cameras.json"
    with open(json_path, "w") as f:
        json.dump(all_cameras, f, indent=4)
        
    print(f"\n✅ Done! Results saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()