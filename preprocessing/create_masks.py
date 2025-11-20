#!/usr/bin/env python
"""
create_masks.py

Folder structure:
  sketches/
    0/
      plain.png
      seat.png
      wheel.png
      ...
    1/
      plain.png
      engine.png
      ...

For each folder N, this script:
  - Loads plain.png (the original sketch)
  - For every other *.png (label image with colored overlay),
    computes the difference from plain.png to get a mask.
  - Saves mask images into a "masks" subfolder as mask_{label}.png
"""

from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image


ROOT_DIR = Path("sketches")   # change if needed
DIFF_THRESH = 20              # bigger => stricter mask (fewer pixels)


def extract_mask(
    plain_img: Image.Image,
    labeled_img: Image.Image,
    diff_thresh: int = DIFF_THRESH
) -> Image.Image:
    """Return a binary mask (PIL L image) of where labeled_img differs from plain_img."""
    # Ensure same size
    if plain_img.size != labeled_img.size:
        raise ValueError(f"Image sizes do not match: {plain_img.size} vs {labeled_img.size}")

    arr_plain = np.asarray(plain_img.convert("RGB"), dtype=np.int16)
    arr_label = np.asarray(labeled_img.convert("RGB"), dtype=np.int16)

    # Absolute RGB difference
    diff = np.abs(arr_label - arr_plain)
    diff_mag = diff.sum(axis=2)  # shape (H, W)

    # Binary mask: 255 where difference is large enough, else 0
    mask = (diff_mag > diff_thresh).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def process_folder(folder: Path) -> None:
    """Process one folder like sketches/0."""
    plain_path = folder / "plain.png"
    if not plain_path.is_file():
        print(f"[skip] {folder} has no plain.png")
        return

    plain_img = Image.open(plain_path)

    # Output directory for masks
    masks_dir = folder / "masks"
    masks_dir.mkdir(exist_ok=True)

    # All label images: *.png except plain.png
    for img_path in sorted(folder.glob("*.png")):
        if img_path.name == "plain.png":
            continue

        label_name = img_path.stem  # e.g. "seat"
        labeled_img = Image.open(img_path)

        try:
            mask_img = extract_mask(plain_img, labeled_img)
        except ValueError as e:
            print(f"[warn] {img_path}: {e}")
            continue

        out_path = masks_dir / f"mask_{label_name}.png"
        mask_img.save(out_path)
        print(f"[ok] {out_path}")


def main() -> None:
    if not ROOT_DIR.is_dir():
        raise SystemExit(f"Root directory {ROOT_DIR} not found")

    # Go through all subfolders (0,1,2,... or any directory)
    for sub in sorted(ROOT_DIR.iterdir()):
        if sub.is_dir():
            print(f"=== Processing {sub} ===")
            process_folder(sub)


if __name__ == "__main__":
    main()
