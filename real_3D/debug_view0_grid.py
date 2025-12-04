#!/usr/bin/env python3
"""
verify_and_overlay.py

1. Reads 'independent_view_fits/all_cameras.json' and the .ply file.
2. Re-renders the views (verification step) to get the base object shape.
3. Looks for "./segmentation/view_{x}/" relative to where this script is run.
4. Overlays masks (e.g., "backrest_1_mask.png") with unique colors.
"""

import json
import math
import numpy as np
import cv2
import open3d as o3d
from open3d.visualization import rendering
from pathlib import Path
import sys

# ---------------- CONFIGURATION ----------------

# Relative paths (assuming you run this script from the root folder)
PLY_PATH       = Path("trellis_outputs/0_trellis_gaussian.ply")
JSON_PATH      = Path("independent_view_fits/all_cameras.json")
SEG_DIR        = Path("segmentations") 
OUTPUT_DIR     = Path("final_overlays")
IMG_SIZE       = 160

# ---------------- COLOR MANAGEMENT ----------------

# Explicit bright colors for visibility
DISTINCT_COLORS = [
    (0, 0, 255),   # Red
    (0, 255, 0),   # Green
    (255, 0, 0),   # Blue
    (0, 255, 255), # Yellow
    (255, 0, 255), # Magenta
    (255, 255, 0), # Cyan
    (0, 165, 255), # Orange
    (128, 0, 128), # Purple
    (0, 128, 128), # Teal
    (128, 128, 0), # Olive
    (128, 128, 128) # Grey
]

object_color_map = {}
color_index = 0

def get_color_for_object(obj_name):
    """Assigns a consistent color to an object name (e.g., 'backrest')."""
    global color_index
    if obj_name not in object_color_map:
        object_color_map[obj_name] = DISTINCT_COLORS[color_index % len(DISTINCT_COLORS)]
        color_index += 1
    return object_color_map[obj_name]

# ---------------- HELPER FUNCTIONS (Rendering) ----------------

def sph_dir(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = math.cos(el) * math.sin(az)
    y = math.cos(el) * math.cos(az)
    z = math.sin(el)
    return np.array([x, y, z], dtype=float)

def load_geometry(path: Path):
    if not path.exists():
        print(f"❌ Error: PLY file not found at {path.resolve()}")
        sys.exit(1)
    geom = o3d.io.read_point_cloud(str(path))
    if geom.is_empty():
        geom = o3d.io.read_triangle_mesh(str(path))
        geom.compute_vertex_normals()
    return geom

def normalize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Centers and scales the mask (85% fill) - same as search script."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    h_box = y_max - y_min + 1
    w_box = x_max - x_min + 1
    crop = mask[y_min:y_max+1, x_min:x_max+1]

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

def render_solid_silhouette(renderer, center, radius, az, el) -> np.ndarray:
    """Renders 3D object from specific view."""
    scene = renderer.scene
    direction = sph_dir(az, el)
    eye = center + radius * direction
    up = np.array([0.0, 0.0, 1.0])
    
    cam = scene.camera
    cam.set_projection(40.0, 1.0, radius*0.01, radius*100.0, rendering.Camera.FovType.Vertical)
    cam.look_at(center, eye, up)
    
    img_o3d = renderer.render_to_image()
    img = np.asarray(img_o3d)
    mask = (img[:, :, 0] < 200).astype(np.uint8)
    return mask

# ---------------- OVERLAY LOGIC ----------------

def parse_filename(filename):
    """
    Input: "backrest_1_mask" (no extension)
    Output: "backrest"
    
    Logic:
    1. Remove '_mask' suffix.
    2. Split by last underscore to remove the count.
    """
    if filename.endswith("_mask"):
        base = filename[:-5]
    else:
        return None 
        
    parts = base.rsplit('_', 1) # Split from right, max 1 split
    if len(parts) == 2:
        return parts[0]
    return base # Fallback

def apply_segmentation_overlays(base_render_bgr, view_name):
    # Construct path: ./segmentation/view_X
    current_seg_dir = SEG_DIR / view_name
    
    if not current_seg_dir.exists():
        print(f"   ⚠️ Warning: Folder not found: {current_seg_dir}")
        return base_render_bgr

    overlay_accum = base_render_bgr.copy()
    
    # Get all mask files
    mask_files = sorted(list(current_seg_dir.glob("*_mask.png")))
    
    if not mask_files:
        print(f"   ℹ️ No masks in {current_seg_dir}")
        return base_render_bgr

    for mask_path in mask_files:
        obj_name = parse_filename(mask_path.stem)
        if not obj_name: continue

        color = get_color_for_object(obj_name)
        
        seg_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if seg_mask is None: continue
        
        # Resize if necessary
        if seg_mask.shape[:2] != (IMG_SIZE, IMG_SIZE):
             seg_mask = cv2.resize(seg_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Paint the color onto the image
        overlay_accum[seg_mask > 0] = color
        print(f"   + Applied {obj_name} ({mask_path.name})")

    return overlay_accum

# ---------------- MAIN ----------------

def main():
    # Debug: print absolute path so user knows where we are looking
    print(f"--- 📂 Path Check ---")
    print(f"Script running in: {Path.cwd()}")
    print(f"Looking for PLY at: {PLY_PATH.resolve()}")
    print(f"Looking for JSON at: {JSON_PATH.resolve()}")
    print(f"Looking for SEG  at: {SEG_DIR.resolve()}")

    if not JSON_PATH.exists():
        print(f"❌ Error: JSON file not found!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(JSON_PATH, "r") as f:
        camera_data = json.load(f)

    # Setup Renderer
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
    center_pt = (IMG_SIZE//2, IMG_SIZE//2)

    print(f"\n--- 🎨 Processing Views ---")

    for view_name, params in camera_data.items():
        if "error" in params:
            continue

        print(f"Rendering {view_name}...")

        # 1. Base Render (The shape)
        az, el, roll = params['azimuth'], params['elevation'], params['roll']
        raw_mask = render_solid_silhouette(renderer, center, radius, az, el)
        norm_mask = normalize_mask(raw_mask, IMG_SIZE)
        M = cv2.getRotationMatrix2D(center_pt, roll, 1.0)
        final_render_mask = cv2.warpAffine(
            norm_mask, M, (IMG_SIZE, IMG_SIZE), 
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        # Create base image (Dark Grey object, White Background)
        base_img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
        base_img[final_render_mask > 0] = [60, 60, 60] 

        # 2. Overlay
        final_img = apply_segmentation_overlays(base_img, view_name)

        # Save
        out_path = OUTPUT_DIR / f"{view_name}_overlaid.png"
        cv2.imwrite(str(out_path), final_img)

    print(f"\n✅ Success! Images saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()