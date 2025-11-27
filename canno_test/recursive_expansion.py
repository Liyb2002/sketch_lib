#!/usr/bin/env python3
"""
expand_recursive_6x6.py — Recursive Zero123++ Generation (Batch Mode)
- Target: All images in sketches/chairs/
- Logic: For each sketch: Generate 6 views -> For each view, generate 6 more -> Total 36
- Config: White Padding, Threshold 160, 2x3 Cut
"""

from pathlib import Path
import shutil
import numpy as np
import torch
import os
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

MODEL_ID = "sudo-ai/zero123plus-v1.2"
CUSTOM_PIPELINE = "sudo-ai/zero123plus-pipeline"

# Input Folder (Processes all .png/.jpg inside)
INPUT_FOLDER = "sketches/chairs/"

# Base Output Folder (Subfolders will be created per object ID)
OUTPUT_BASE = "sketches/chairs/expanded_36/"

# Grid Layout: 2 Columns, 3 Rows
N_COLS = 2
N_ROWS = 3

# Inference Steps
NUM_STEPS = 50

# Input resolution
IMG_SIZE = 320 

# Background Cleaning Threshold (0-255)
BG_THRESHOLD = 160

# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

def prepare_input_image(image_input, size: int = IMG_SIZE) -> Image.Image:
    """
    Accepts Path or PIL Image. Pads to square on WHITE background.
    """
    if isinstance(image_input, (str, Path)):
        if not Path(image_input).exists():
            raise FileNotFoundError(f"Input image not found: {image_input}")
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    w, h = img.size
    side = max(w, h)
    
    # Use WHITE background for input padding as requested
    bg = Image.new("RGB", (side, side), (255, 255, 255))
    bg.paste(img, ((side - w) // 2, (side - h) // 2))
    
    return bg.resize((size, size), Image.BICUBIC)


def clean_background(img: Image.Image, threshold: int = BG_THRESHOLD) -> Image.Image:
    """Convert grey/light background pixels to Pure White."""
    arr = np.array(img).astype(np.uint8)
    gray = arr.mean(axis=2)
    mask_bg = gray > threshold
    out = arr.copy()
    out[mask_bg] = [255, 255, 255] # Force to White
    return Image.fromarray(out)


def crop_grid_to_views(grid_img: Image.Image, n_cols: int, n_rows: int):
    """Split a grid image into n_cols * n_rows tiles."""
    w, h = grid_img.size
    tile_w = w // n_cols
    tile_h = h // n_rows
    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            box = (c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)
            tiles.append(grid_img.crop(box))
    return tiles


def main():
    input_root = ROOT / INPUT_FOLDER
    output_root = ROOT / OUTPUT_BASE

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    # 1. LOAD MODEL
    print(f"[NVS] Loading model: {MODEL_ID}")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID, 
        custom_pipeline=CUSTOM_PIPELINE,
        torch_dtype=torch.float16
    )
    try:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    except Exception:
        pass
    pipe.to("cuda")

    # 2. FIND IMAGES
    valid_exts = {".png", ".jpg", ".jpeg"}
    image_files = sorted([
        f for f in input_root.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_exts
    ])

    print(f"[NVS] Found {len(image_files)} images in {input_root}")

    # 3. PROCESS LOOP
    for target_path in image_files:
        sketch_id = target_path.stem # e.g., "0" from "0.png"
        object_output_dir = output_root / sketch_id
        
        # Skip if already done (check for 36 images)
        if object_output_dir.exists() and len(list(object_output_dir.glob("*.png"))) >= 36:
            print(f"Skipping {sketch_id} (Already processed)")
            continue

        # Setup output folder for this object
        if object_output_dir.exists():
            shutil.rmtree(object_output_dir)
        object_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[NVS] Processing: {target_path.name} -> {object_output_dir}")

        # --- ROUND 1: GENERATE PARENT VIEWS ---
        print("  > Round 1: Generating initial 6 views...")
        input_img = prepare_input_image(target_path)
        
        with torch.autocast("cuda"):
            base_grid = pipe(input_img, num_inference_steps=NUM_STEPS).images[0]
        
        parent_views = crop_grid_to_views(base_grid, N_COLS, N_ROWS)
        
        # --- ROUND 2: RECURSIVE EXPANSION ---
        image_counter = 0
        
        for p_idx, parent_view in enumerate(parent_views):
            print(f"  > Round 2: Expanding Parent View {p_idx}...")
            
            # Prepare parent view as new input (Pad to white square again)
            child_input = prepare_input_image(parent_view)
            
            with torch.autocast("cuda"):
                child_grid = pipe(child_input, num_inference_steps=NUM_STEPS).images[0]
                
            child_views = crop_grid_to_views(child_grid, N_COLS, N_ROWS)
            
            # Clean and Save
            for c_idx, child_view in enumerate(child_views):
                cleaned = clean_background(child_view, threshold=BG_THRESHOLD)
                
                # Filename: linear count 0.png ... 35.png
                save_name = f"{image_counter}.png"
                cleaned.save(object_output_dir / save_name)
                
                image_counter += 1

        print(f"  > Saved {image_counter} views for {sketch_id}")

    print(f"\n[NVS] Batch Expansion Complete!")

if __name__ == "__main__":
    main()