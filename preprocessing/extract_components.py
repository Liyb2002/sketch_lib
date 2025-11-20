#!/usr/bin/env python
"""
extract_components.py  (formerly create_components.py)

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
    compares it with plain.png to find the overlay region,
    merges nearby pixels (morphological closing) so that each
    semantic part (e.g., each wheel) is one big blob,
    runs connected components, filters out tiny blobs,
    and for each big component:
      * applies its mask to plain.png,
      * crops to the component's bounding box,
      * saves it as:

          components/component_{label_name}_{k}.png
          (RGBA, transparent background)

    e.g. if wheel.png has two wheel regions:
      components/component_wheel_1.png
      components/component_wheel_2.png
"""

from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, label as cc_label  # <-- alias label


# ----------------- config -----------------
ROOT_DIR   = Path("sketches")  # change if needed
DIFF_THRESH = 20               # diff threshold to detect overlay pixels
CLOSE_SIZE  = 25               # size of morphological closing kernel
MIN_AREA    = 5000             # ignore components smaller than this (in pixels)


# ----------------- helpers -----------------
def compute_diff_mask(plain: Image.Image,
                      overlay: Image.Image,
                      diff_thresh: int = DIFF_THRESH) -> np.ndarray:
    """
    Compute a binary mask where overlay differs from plain beyond diff_thresh.
    Returns a bool/uint8 array of shape (H, W).
    """
    if plain.size != overlay.size:
        raise ValueError(f"Image sizes do not match: {plain.size} vs {overlay.size}")

    arr_plain = np.asarray(plain.convert("RGB"), dtype=np.int16)
    arr_over  = np.asarray(overlay.convert("RGB"), dtype=np.int16)

    diff = np.abs(arr_plain - arr_over)
    diff_mag = diff.sum(axis=2)
    mask = diff_mag > diff_thresh
    return mask.astype(np.uint8)


def merge_fragments(mask: np.ndarray,
                    close_size: int = CLOSE_SIZE) -> np.ndarray:
    """
    Apply morphological closing to merge small gaps so large components
    (like each wheel) become one connected region.
    """
    structure = np.ones((close_size, close_size), dtype=np.uint8)
    merged = binary_closing(mask, structure=structure)
    return merged.astype(np.uint8)


def extract_component(plain_img: Image.Image,
                      comp_mask: np.ndarray,
                      pad: int = 2) -> Optional[Image.Image]:
    """
    Given the plain sketch and a binary mask (0/1) of ONE component,
    return a cropped RGBA image containing only that component on a
    transparent background. Returns None if mask is empty.
    """
    ys, xs = np.where(comp_mask == 1)
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

    # Transparent canvas
    comp_arr = np.zeros_like(arr_plain)
    comp_arr[..., 3] = 0

    # Copy only masked pixels
    comp_arr[comp_mask == 1] = arr_plain[comp_mask == 1]

    # Crop to bounding box
    comp_crop = comp_arr[y_min:y_max + 1, x_min:x_max + 1]
    return Image.fromarray(comp_crop, mode="RGBA")


# ----------------- per-folder processing -----------------
def process_folder(folder: Path) -> None:
    plain_path = folder / "plain.png"
    if not plain_path.is_file():
        print(f"[skip] {folder} has no plain.png")
        return

    plain_img = Image.open(plain_path)

    # Output directory for components
    comps_dir = folder / "components"
    comps_dir.mkdir(exist_ok=True)

    # All overlay label images except plain.png
    for label_img_path in sorted(folder.glob("*.png")):
        if label_img_path.name == "plain.png":
            continue

        label_name = label_img_path.stem  # e.g. "wheel", "seat"
        overlay_img = Image.open(label_img_path)

        try:
            raw_mask = compute_diff_mask(plain_img, overlay_img)
        except ValueError as e:
            print(f"[warn] {label_img_path}: {e}")
            continue

        # Merge small gaps so each semantic part becomes one blob
        merged_mask = merge_fragments(raw_mask)

        # Connected components
        labeled, num = cc_label(merged_mask)
        if num == 0:
            print(f"[warn] {label_img_path}: no components after closing")
            continue

        comp_index = 1
        for comp_id in range(1, num + 1):
            comp_mask = (labeled == comp_id).astype(np.uint8)

            area = comp_mask.sum()
            if area < MIN_AREA:
                continue  # ignore tiny fragments

            comp_img = extract_component(plain_img, comp_mask)
            if comp_img is None:
                continue

            out_path = comps_dir / f"component_{label_name}_{comp_index}.png"
            comp_img.save(out_path)
            print(f"[ok] {out_path}")
            comp_index += 1


# ----------------- main -----------------
def main() -> None:
    if not ROOT_DIR.is_dir():
        raise SystemExit(f"Root directory {ROOT_DIR} not found")

    for folder in sorted(ROOT_DIR.iterdir()):
        if folder.is_dir():
            print(f"=== Processing {folder} ===")
            process_folder(folder)


if __name__ == "__main__":
    main()
