#!/usr/bin/env python3
"""
generate_views_flux2_zero123.py — run Zero123++ multi-view generation
for a sketch and its corresponding realistic version, then applies
a luminance threshold cleaning to remove grey backgrounds.

Input:
    - sketch/input.png
    - sketch/input_realistic.png

Outputs (6 views each with clean white backgrounds):
    - sketch/views/view_0.png ... view_5.png
    - sketch/views_realistic/view_0.png ... view_5.png

NOTE: The pipeline output is a grid internally, but we DO NOT save the grid.
We only save the cropped view tiles.
"""

from pathlib import Path
from typing import List

import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# ---------------------------------------------------------------------
# CONFIG: Zero123++ model
# ---------------------------------------------------------------------

MODEL_ID = "sudo-ai/zero123plus-v1.2"
CUSTOM_PIPELINE = "sudo-ai/zero123plus-pipeline"

# Grid Configuration: 2 columns x 3 rows
N_COLS = 2
N_ROWS = 3
N_VIEWS = N_COLS * N_ROWS

IMG_SIZE = 320  # input size for conditioning image (Zero123++ standard)

# Background cleaning threshold (average channel value 0-255)
LUMINANCE_THRESHOLD = 160

# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SKETCH_FOLDER = ROOT / "sketch"

INPUT_SKETCH_PATH = SKETCH_FOLDER / "input.png"
INPUT_REALISTIC_PATH = SKETCH_FOLDER / "input_realistic.png"
OUTPUT_SKETCH_VIEWS_DIR = SKETCH_FOLDER / "views"
OUTPUT_REALISTIC_VIEWS_DIR = SKETCH_FOLDER / "views_realistic"


def load_square_image(path: Path, size: int = IMG_SIZE) -> Image.Image:
    """Load image, pad to square on pure white background, resize to (size, size)."""
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    img = Image.open(path)

    # Handle transparency and ensure RGB
    if img.mode == "RGBA":
        *rgb, alpha = img.split()
        bg = Image.new("RGB", img.size, "white")
        bg.paste(Image.merge("RGB", rgb), mask=alpha)
        img = bg
    else:
        img = img.convert("RGB")

    # Pad to square
    w, h = img.size
    side = max(w, h)
    square_bg = Image.new("RGB", (side, side), "white")
    square_bg.paste(img, ((side - w) // 2, (side - h) // 2))

    # Resize
    return square_bg.resize((size, size), Image.Resampling.LANCZOS)


def crop_grid_to_views(grid_img: Image.Image, n_cols: int, n_rows: int) -> List[Image.Image]:
    """Split a grid image into n_cols * n_rows tiles (row-major order)."""
    w, h = grid_img.size
    tile_w = w // n_cols
    tile_h = h // n_rows
    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            box = (c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)
            tiles.append(grid_img.crop(box))
    return tiles


def clean_background(tile: Image.Image, threshold: int = LUMINANCE_THRESHOLD) -> Image.Image:
    """Convert light grey/noisy background pixels to pure white using luminance threshold."""
    arr = np.array(tile).astype(np.uint8)
    gray = arr.mean(axis=2)
    mask_bg = gray > threshold
    out = arr.copy()
    out[mask_bg] = [255, 255, 255]
    return Image.fromarray(out)


def generate_views(pipe: DiffusionPipeline, input_path: Path, output_dir: Path):
    """Generate 6 views, clean backgrounds, and save ONLY the cropped tiles."""
    print(f"\n[Zero123++] Starting generation for: {input_path.name}")

    try:
        cond_img = load_square_image(input_path, size=IMG_SIZE)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}. Skipping this input.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Running NVS pipeline...")
    with torch.autocast("cuda"):
        out = pipe(cond_img, num_inference_steps=50)

    # Extract grid image from pipeline output
    if hasattr(out, "images"):
        grid = out.images[0]
    elif isinstance(out, list):
        grid = out[0]
    else:
        raise RuntimeError("Unexpected pipeline output format from Zero123++.")

    # Crop (internally) and save only tiles
    views = crop_grid_to_views(grid, N_COLS, N_ROWS)
    print(f"  Saving {N_VIEWS} CLEANED cropped views to: {output_dir}")

    for i, tile in enumerate(views):
        cleaned = clean_background(tile)
        view_path = output_dir / f"view_{i}.png"
        cleaned.save(view_path)

    print(f"  Done: {output_dir}")


def main():
    if not SKETCH_FOLDER.is_dir():
        print(f"[ERROR] Base sketch folder not found: {SKETCH_FOLDER}. Creating it.")
        SKETCH_FOLDER.mkdir(exist_ok=True)

    if not INPUT_SKETCH_PATH.exists() or not INPUT_REALISTIC_PATH.exists():
        print(
            f"[ERROR] Required inputs ({INPUT_SKETCH_PATH.name} and "
            f"{INPUT_REALISTIC_PATH.name}) are missing from '{SKETCH_FOLDER.name}'."
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Activate zero123pp_env (or similar) with GPU support.")

    print(f"[Zero123++] Loading model: {MODEL_ID}")
    pipe_kwargs = {
        "pretrained_model_name_or_path": MODEL_ID,
        "torch_dtype": torch.float16,
    }
    if CUSTOM_PIPELINE is not None:
        pipe_kwargs["custom_pipeline"] = CUSTOM_PIPELINE

    pipe = DiffusionPipeline.from_pretrained(**pipe_kwargs)

    # Optional scheduler swap
    try:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    except Exception as e:
        print(f"[Zero123++] Could not switch scheduler: {e}. Using default.")

    pipe.to("cuda")
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    generate_views(pipe=pipe, input_path=INPUT_SKETCH_PATH, output_dir=OUTPUT_SKETCH_VIEWS_DIR)
    generate_views(pipe=pipe, input_path=INPUT_REALISTIC_PATH, output_dir=OUTPUT_REALISTIC_VIEWS_DIR)

    print("\n[Zero123++] All done. Only cropped view tiles were saved.")


if __name__ == "__main__":
    main()
