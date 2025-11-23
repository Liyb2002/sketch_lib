#!/usr/bin/env python3
"""
generate_views.py — Zero123++ multi-view generation via sudo-ai/zero123plus-pipeline

Input:
    sketches/0/plain.png

Output:
    sketches/0/views/zero123pp_grid.png
    sketches/0/views/view_0.png ... view_5.png
"""

from pathlib import Path

import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler


ROOT = Path(__file__).resolve().parent
INPUT_IMG = ROOT / "sketches" / "0" / "plain.png"
OUTPUT_DIR = ROOT / "sketches" / "0" / "views"
GRID_OUT = OUTPUT_DIR / "zero123pp_grid.png"


def load_square_image(path: Path, size: int = 320) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w == h:
        sq = img
    else:
        side = max(w, h)
        bg = Image.new("RGB", (side, side), (255, 255, 255))
        offset = ((side - w) // 2, (side - h) // 2)
        bg.paste(img, offset)
        sq = bg

    if size is not None:
        sq = sq.resize((size, size), Image.BICUBIC)
    return sq


def crop_grid_to_views(grid_img: Image.Image, n_cols: int = 3, n_rows: int = 2):
    w, h = grid_img.size
    tile_w = w // n_cols
    tile_h = h // n_rows

    views = []
    for row in range(n_rows):
        for col in range(n_cols):
            left = col * tile_w
            upper = row * tile_h
            right = left + tile_w
            lower = upper + tile_h
            crop = grid_img.crop((left, upper, right, lower))
            views.append(crop)
    return views


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Activate zero123pp_env with GPU support.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Zero123++] Loading input: {INPUT_IMG}")
    cond_img = load_square_image(INPUT_IMG, size=320)

    print("[Zero123++] Loading pipeline sudo-ai/zero123plus-v1.2 with custom pipeline sudo-ai/zero123plus-pipeline ...")
    pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.to("cuda")

    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    print("[Zero123++] Generating multi-view grid...")
    with torch.autocast("cuda"):
        result = pipe(cond_img, num_inference_steps=50).images[0]

    print(f"[Zero123++] Saving grid to: {GRID_OUT}")
    result.save(GRID_OUT)

    print("[Zero123++] Cropping grid into individual views...")
    views = crop_grid_to_views(result, n_cols=3, n_rows=2)

    for i, v in enumerate(views):
        out_path = OUTPUT_DIR / f"view_{i}.png"
        v.save(out_path)
        print(f"  saved {out_path}")

    print("[Zero123++] Done.")


if __name__ == "__main__":
    main()
