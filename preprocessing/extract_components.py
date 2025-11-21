#!/usr/bin/env python
"""
create_components.py

Assumed folder structure:
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
    computes a mask via diff, applies it to plain.png,
    crops to the component's bounding box, and saves it as:
        components/component_{label}.png  (RGBA, transparent background)
"""

from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image


ROOT_DIR = Path("sketches")   # change if needed
DIFF_THRESH = 20              # same as in create_masks.py


def compute_mask(plain_img: Image.Image,
                 labeled_img: Image.Image,
                 diff_thresh: int = DIFF_THRESH) -> np.ndarray:
    """
    Compute a binary mask (H, W) where labeled_img differs from plain_img.
    Returns a uint8 array with values 0 or 1.
    """
    if plain_img.size != labeled_img.size:
        raise ValueError(f"Image sizes do not match: {plain_img.size} vs {labeled_img.size}")

    arr_plain = np.asarray(plain_img.convert("RGB"), dtype=np.int16)
    arr_label = np.asarray(labeled_img.convert("RGB"), dtype=np.int16)

    diff = np.abs(arr_label - arr_plain)
    diff_mag = diff.sum(axis=2)  # (H, W)
    mask = (diff_mag > diff_thresh).astype(np.uint8)
    return mask


def extract_component(
    plain_img: Image.Image,
    mask: np.ndarray,
    pad: int = 2
) -> Image.Image:
    """
    Given the plain sketch and a binary mask (0/1) of the component,
    return a cropped RGBA image containing only that component on a
    transparent background.
    """
    # If mask is empty, just return None
    ys, xs = np.where(mask == 1)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Add a little padding, clamped to image bounds
    w, h = plain_img.size
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w - 1, x_max + pad)
    y_max = min(h - 1, y_max + pad)

    # Convert plain image to RGBA
    plain_rgba = plain_img.convert("RGBA")
    arr_plain = np.asarray(plain_rgba)

    # Create transparent canvas
    comp_arr = np.zeros_like(arr_plain)
    comp_arr[..., 3] = 0  # alpha channel = 0 (transparent) everywhere

    # Copy only masked pixels
    comp_arr[mask == 1] = arr_plain[mask == 1]

    # Crop to bounding box
    comp_arr_cropped = comp_arr[y_min:y_max + 1, x_min:x_max + 1]

    return Image.fromarray(comp_arr_cropped, mode="RGBA")


def process_folder(folder: Path) -> None:
    plain_path = folder / "plain.png"
    if not plain_path.is_file():
        print(f"[skip] {folder} has no plain.png")
        return

    plain_img = Image.open(plain_path)

    # Output directory for components
    comps_dir = folder / "components"
    comps_dir.mkdir(exist_ok=True)

    # All label images except plain.png
    for img_path in sorted(folder.glob("*.png")):
        if img_path.name == "plain.png":
            continue

        label_name = img_path.stem  # e.g. "seat"
        labeled_img = Image.open(img_path)

        try:
            mask = compute_mask(plain_img, labeled_img)
        except ValueError as e:
            print(f"[warn] {img_path}: {e}")
            continue

        component_img = extract_component(plain_img, mask)
        if component_img is None:
            print(f"[warn] {img_path}: empty mask, skipping")
            continue

        out_path = comps_dir / f"component_{label_name}.png"
        component_img.save(out_path)
        print(f"[ok] {out_path}")


def main() -> None:
    if not ROOT_DIR.is_dir():
        raise SystemExit(f"Root directory {ROOT_DIR} not found")

    for sub in sorted(ROOT_DIR.iterdir()):
        if sub.is_dir():
            print(f"=== Processing {sub} ===")
            process_folder(sub)


if __name__ == "__main__":
    main()
