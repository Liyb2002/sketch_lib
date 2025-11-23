#!/usr/bin/env python3
"""
generate_views.py — generic multi-view generation for a Zero123-style model.

You only need to edit:
    MODEL_ID
    CUSTOM_PIPELINE (optional)
    N_COLS, N_ROWS (grid layout)
"""

from pathlib import Path
import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# ---------------------------------------------------------------------
# CONFIG: put your chosen HF model ID here
# ---------------------------------------------------------------------

# Example that we KNOW worked for you:
# MODEL_ID = "sudo-ai/zero123plus-v1.2"
# CUSTOM_PIPELINE = "sudo-ai/zero123plus-pipeline"

MODEL_ID = "sudo-ai/zero123plus-v1.2"
CUSTOM_PIPELINE = "sudo-ai/zero123plus-pipeline"  # or None for standard diffusers models

# Grid layout of the output image:
# For zero123++ it's usually 3x2 (6 views).
N_COLS = 3
N_ROWS = 2

# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
INPUT_IMG = ROOT / "sketches" / "0" / "plain.png"
OUTPUT_DIR = ROOT / "sketches" / "0" / "views"
GRID_OUT = OUTPUT_DIR / "zero123_grid.png"


def load_square_image(path: Path, size: int = 320) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = max(w, h)
    bg = Image.new("RGB", (side, side), "white")
    bg.paste(img, ((side - w) // 2, (side - h) // 2))
    return bg.resize((size, size), Image.BICUBIC)


def crop_grid_to_views(grid_img: Image.Image, n_cols: int, n_rows: int):
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Activate zero123pp_env with GPU support.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[NVS] Loading input: {INPUT_IMG}")
    cond_img = load_square_image(INPUT_IMG, size=320)

    print(f"[NVS] Loading model: {MODEL_ID}")
    pipe_kwargs = {
        "pretrained_model_name_or_path": MODEL_ID,
        "torch_dtype": torch.float16,
    }
    if CUSTOM_PIPELINE is not None:
        pipe_kwargs["custom_pipeline"] = CUSTOM_PIPELINE

    pipe = DiffusionPipeline.from_pretrained(**pipe_kwargs)

    # Try to use EulerAncestral; if unsupported, keep default
    try:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    except Exception as e:
        print(f"[NVS] Could not switch scheduler to EulerAncestral: {e}")
        print("[NVS] Using default scheduler.")

    pipe.to("cuda")
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    print("[NVS] Generating multi-view grid...")
    with torch.autocast("cuda"):
        out = pipe(cond_img, num_inference_steps=50)

    # diffusers returns an object with .images in most pipelines
    if hasattr(out, "images"):
        grid = out.images[0]
    elif isinstance(out, list):
        grid = out[0]
    else:
        raise RuntimeError("Unexpected pipeline output format.")

    print(f"[NVS] Saving grid to: {GRID_OUT}")
    grid.save(GRID_OUT)

    print(f"[NVS] Cropping grid into {N_COLS * N_ROWS} views...")
    views = crop_grid_to_views(grid, N_COLS, N_ROWS)
    for i, v in enumerate(views):
        outpath = OUTPUT_DIR / f"view_{i}.png"
        v.save(outpath)
        print(f"  saved {outpath}")

    print("[NVS] Done.")


if __name__ == "__main__":
    main()
