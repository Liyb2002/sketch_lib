#!/usr/bin/env python3
"""
fast_pose_search.py

Strategy:
1. Coarse Search (Global):
   - Azimuth/Roll Step: 20°
   - Elevation Step: 20°
   - Uses fast 2D rotations (cv2) to check ~3,000 poses quickly.
   
2. Fine Search (Local):
   - Takes ONLY the #1 best result from Coarse search.
   - Refines with Step 1° in a +/- 10° range.
   - 3D Renders: ~440 (Manageable speed).

Output: 'pose_search' folder with results.
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
SKETCH_PATH = Path("0.png")
OUT_DIR     = Path("pose_search")

IMG_SIZE = 256

# ---------------- HELPER FUNCTIONS ----------------

def sph_dir(az_deg, el_deg):
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

def make_unlit_black_material():
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.base_color = (0.0, 0.0, 0.0, 1.0)
    mat.point_size = 2.5 
    return mat

def normalize_mask(mask: np.ndarray, size: int, margin: float = 0.1) -> np.ndarray:
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
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Sketch not found: {path}")
        
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if np.mean(img) > 127: img = 255 - img
    _, mask = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(mask)
    cv2.drawContours(sil, contours, -1, 255, thickness=-1)
    
    return normalize_mask((sil > 0).astype(np.uint8), size)

def render_solid_silhouette(renderer, center, radius, az, el) -> np.ndarray:
    scene = renderer.scene
    direction = sph_dir(az, el)
    eye = center + radius * direction
    up = np.array([0.0, 0.0, 1.0])
    
    cam = scene.camera
    cam.set_projection(40.0, 1.0, radius*0.1, radius*10.0, rendering.Camera.FovType.Vertical)
    cam.look_at(center, eye, up)
    
    img_o3d = renderer.render_to_image()
    img = np.asarray(img_o3d)
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


# ---------------- MAIN ----------------

def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    geom = load_geometry(PLY_PATH)
    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    radius = 2.5 * float(np.max(bbox.get_extent()))
    sketch_mask = load_sketch_mask(SKETCH_PATH, IMG_SIZE)
    
    # 2. Setup Renderer
    renderer = rendering.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.add_geometry("obj", geom, make_unlit_black_material())

    # ==========================================
    # STAGE 1: COARSE SEARCH (Global, Step 20)
    # ==========================================
    
    AZ_COARSE = np.arange(0, 360, 20)
    EL_COARSE = np.arange(-80, 81, 20)
    ROLL_COARSE = np.arange(0, 360, 20)
    
    # Pre-calculate Roll Matrices (Optimization)
    roll_matrices = {}
    center_pt = (IMG_SIZE//2, IMG_SIZE//2)
    for r in ROLL_COARSE:
        roll_matrices[r] = cv2.getRotationMatrix2D(center_pt, r, 1.0)
        
    best_coarse_score = -1.0
    best_coarse_params = (0, 0, 0)
    
    # Progress Bar: Tracks Az*El steps (rendering steps)
    total_renders = len(AZ_COARSE) * len(EL_COARSE)
    pbar_c = tqdm(total=total_renders, desc="Stage 1: Coarse Search")
    
    for az, el in itertools.product(AZ_COARSE, EL_COARSE):
        # 1. Render 3D (Expensive)
        raw_mask = render_solid_silhouette(renderer, center, radius, az, el)
        norm_mask = normalize_mask(raw_mask, IMG_SIZE)
        
        # 2. Rotate 2D (Cheap)
        for roll in ROLL_COARSE:
            M = roll_matrices[roll]
            rolled_mask = cv2.warpAffine(
                norm_mask, M, (IMG_SIZE, IMG_SIZE), 
                flags=cv2.INTER_NEAREST, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
            score = get_iou(sketch_mask, rolled_mask)
            
            if score > best_coarse_score:
                best_coarse_score = score
                best_coarse_params = (az, el, roll)
        
        pbar_c.update(1)
    pbar_c.close()
    
    c_az, c_el, c_roll = best_coarse_params
    print(f"   Best Coarse Match: IoU={best_coarse_score:.4f} at (Az {c_az}, El {c_el}, Roll {c_roll})")


    # ==========================================
    # STAGE 2: FINE SEARCH (Local, Step 1)
    # ==========================================
    
    # Range: +/- 10 degrees around the single best coarse result
    az_range   = range(int(c_az - 15), int(c_az + 16))
    el_range   = range(int(c_el - 15), int(c_el + 16))
    roll_range = range(int(c_roll - 15), int(c_roll + 16))
    
    best_fine_score = -1.0
    best_fine_params = None
    best_fine_mask = None
    
    # Total operations for progress bar
    total_fine = len(az_range) * len(el_range)
    pbar_f = tqdm(total=total_fine, desc="Stage 2: Fine Search (+/- 10 deg)")
    
    for f_az in az_range:
        for f_el in el_range:
            
            # Clamp elevation to physical limits
            render_el = max(-89, min(89, f_el))
            render_az = f_az % 360 # Wrap azimuth
            
            # 1. Render 3D (Expensive, but fewer calls now)
            raw_mask = render_solid_silhouette(renderer, center, radius, render_az, render_el)
            norm_mask = normalize_mask(raw_mask, IMG_SIZE)
            
            # 2. Rotate 2D (Fine steps)
            for f_roll in roll_range:
                render_roll = f_roll % 360
                
                # Compute matrix on fly (since range is dynamic)
                M = cv2.getRotationMatrix2D(center_pt, render_roll, 1.0)
                rolled_mask = cv2.warpAffine(
                    norm_mask, M, (IMG_SIZE, IMG_SIZE), 
                    flags=cv2.INTER_NEAREST, 
                    borderMode=cv2.BORDER_CONSTANT, 
                    borderValue=0
                )
                
                score = get_iou(sketch_mask, rolled_mask)
                
                if score > best_fine_score:
                    best_fine_score = score
                    best_fine_params = (render_az, render_el, render_roll)
                    best_fine_mask = rolled_mask.copy()
            
            pbar_f.update(1)
            
    pbar_f.close()


    # ==========================================
    # SAVE OUTPUT
    # ==========================================
    
    final_az, final_el, final_roll = best_fine_params
    print(f"   Best Fine Match:   IoU={best_fine_score:.4f} at (Az {final_az}, El {final_el}, Roll {final_roll})")
    
    # Save Overlay
    save_overlay(OUT_DIR / "best_match_overlay.png", sketch_mask, best_fine_mask)
    
    # Save Render Mask
    cv2.imwrite(str(OUT_DIR / "best_match_render.png"), best_fine_mask * 255)
    
    # Save JSON
    data = {
        "iou_score": float(best_fine_score),
        "azimuth": int(final_az),
        "elevation": int(final_el),
        "roll": int(final_roll)
    }
    with open(OUT_DIR / "best_pose.json", "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"\n✅ Saved best result to {OUT_DIR}")

if __name__ == "__main__":
    main()