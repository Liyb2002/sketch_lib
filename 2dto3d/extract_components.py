#!/usr/bin/env python3
"""
extract_components_clean.py — split Zero123++ 2×3 grid into 6 images
and convert grey backgrounds into pure white while preserving sketch lines.
"""

from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent
SKETCHES_DIR = ROOT / "sketches"

# Your correct grid layout: 2 per row, 3 rows
N_COLS = 2
N_ROWS = 3


def split_grid(img: Image.Image):
    """Split a 2×3 grid into 6 tiles."""
    w, h = img.size
    tile_w = w // N_COLS
    tile_h = h // N_ROWS

    tiles = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            box = (c * tile_w,
                   r * tile_h,
                   (c + 1) * tile_w,
                   (r + 1) * tile_h)
            tiles.append(img.crop(box))
    return tiles


def clean_background(tile: Image.Image) -> Image.Image:
    """
    Convert grey background to pure white while keeping sketch lines.
    Works because sketches are dark & background is light grey.
    """

    arr = np.array(tile).astype(np.uint8)

    # Convert to grayscale to detect background
    gray = arr.mean(axis=2)

    # Threshold: everything lighter than 200 becomes pure white
    # You can adjust 200 → 180 or 220 depending on style
    mask_bg = gray > 165

    # Create a white canvas
    out = arr.copy()
    out[mask_bg] = [255, 255, 255]

    return Image.fromarray(out)


def main():
    object_folders = [p for p in SKETCHES_DIR.iterdir() if p.is_dir()]

    for obj_dir in object_folders:
        print(f"\n[CLEAN] Object: {obj_dir.name}")

        views_dir = obj_dir / "views"
        if not views_dir.exists():
            print("  no views/ folder, skipping")
            continue

        out_dir = obj_dir / "components"
        out_dir.mkdir(exist_ok=True)

        grid_paths = sorted(views_dir.glob("*_grid.png"))
        if not grid_paths:
            print("  no *_grid.png found")
            continue

        for grid_path in grid_paths:
            print(f"  Processing {grid_path.name}")
            grid_img = Image.open(grid_path).convert("RGB")
            tiles = split_grid(grid_img)

            stem = grid_path.stem.replace("_grid", "")

            for i, tile in enumerate(tiles):
                cleaned = clean_background(tile)
                out_path = out_dir / f"{stem}_comp_{i}.png"
                cleaned.save(out_path)
                print(f"    Saved {out_path}")


if __name__ == "__main__":
    main()
