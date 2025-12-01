#!/usr/bin/env python3
"""
expand_single_6.py — Single Zero123++ Generation for 0.png
- Target: 0.png in the same folder as this script
- Logic: Generate 6 views -> Clean -> Save
- Config: White Padding, Threshold 160, 2x3 Cut
- Output: 6 images (0.png - 5.png) in 6_views_test/
"""

from pathlib import Path
import shutil
import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

MODEL_ID = "sudo-ai/zero123plus-v1.2"
CUSTOM_PIPELINE = "sudo-ai/zero123plus-pipeline"

# Input image name (in same folder as this script)
INPUT_IMAGE_NAME = "0.png"

# Output Folder (relative to this script)
OUTPUT_FOLDER = "6_views_test/"

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
    
    # Use WHITE background for input padding
    bg = Image.new("RGB", (side, side), (255, 255, 255))
    bg.paste(img, ((side - w) // 2, (side - h) // 2))
    
    return bg.resize((size, size), Image.BICUBIC)


def clean_background(img: Image.Image, threshold: int = BG_THRESHOLD) -> Image.Image:
    """Convert grey/light background pixels to Pure White."""
    arr = np.array(img).astype(np.uint8)
    gray = arr.mean(axis=2)
    mask_bg = gray > threshold
    out = arr.copy()
    out[mask_bg] = [255, 255, 255]  # Force to White
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
    input_path = ROOT / INPUT_IMAGE_NAME
    output_root = ROOT / OUTPUT_FOLDER

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Recreate the output folder fresh
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

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

    print(f"\n[NVS] Processing single image: {input_path.name} -> {output_root}")

    # --- GENERATE VIEWS ---
    print("  > Generating 6 views...")
    input_img = prepare_input_image(input_path)
    
    with torch.autocast("cuda"):
        base_grid = pipe(input_img, num_inference_steps=NUM_STEPS).images[0]
    
    raw_views = crop_grid_to_views(base_grid, N_COLS, N_ROWS)
    
    # --- CLEAN AND SAVE ---
    count = 0
    for v in raw_views:
        cleaned = clean_background(v, threshold=BG_THRESHOLD)
        save_name = f"{count}.png"
        cleaned.save(output_root / save_name)
        count += 1

    print(f"  > Saved {count} views in {output_root}")
    print(f"\n[NVS] Single Expansion Complete!")

if __name__ == "__main__":
    main()
