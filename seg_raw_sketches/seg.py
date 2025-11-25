#!/usr/bin/env python3
"""
Run SAM on all sketches and save individual objects.

- Root folder: sketches/
- For each subfolder (e.g. chairs/), create chairs/individual/
- For each image, find SAM masks, make 1.2x bbox crops, and
  place the object on white background.
"""

import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ------------- config -------------
ROOT_DIR = Path("sketches")           # root folder of all categories
MODEL_TYPE = "vit_h"                  # "vit_h", "vit_l", or "vit_b"
CHECKPOINT = "sam_vit_h_4b8939.pth"   # path to your SAM checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# mask filtering
MIN_AREA_RATIO = 0.01   # ignore masks smaller than 1% of image area
MAX_MASKS_PER_IMAGE = 30  # safety cap


# ------------- helpers -------------

def load_sam():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
    sam.to(device=DEVICE)
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=0,
    )


def get_expanded_bbox(mask, scale: float, img_w: int, img_h: int):
    """Compute a scale× expanded bbox around a boolean mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    w = x1 - x0 + 1
    h = y1 - y0 + 1
    cx = x0 + w / 2.0
    cy = y0 + h / 2.0

    new_w = w * scale
    new_h = h * scale

    nx0 = int(round(cx - new_w / 2.0))
    ny0 = int(round(cy - new_h / 2.0))
    nx1 = int(round(cx + new_w / 2.0))
    ny1 = int(round(cy + new_h / 2.0))

    nx0 = max(0, nx0)
    ny0 = max(0, ny0)
    nx1 = min(img_w, nx1)
    ny1 = min(img_h, ny1)

    if nx1 <= nx0 or ny1 <= ny0:
        return None
    return nx0, ny0, nx1, ny1


def extract_object(image: Image.Image, mask: np.ndarray, bbox):
    """
    Crop to bbox, put object on white background using mask.
    image: PIL RGB
    mask: HxW bool or 0/1
    bbox: (x0, y0, x1, y1) in image coords
    """
    x0, y0, x1, y1 = bbox
    crop = image.crop((x0, y0, x1, y1))          # RGB crop
    crop_w, crop_h = crop.size

    # crop mask to bbox
    mask_crop = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_crop, mode="L")  # grayscale alpha

    # white background
    white_bg = Image.new("RGB", (crop_w, crop_h), (255, 255, 255))
    # paste object where mask==1
    white_bg.paste(crop, mask=mask_img)

    return white_bg


# ------------- main logic -------------

def process_image(mask_generator, img_path: Path, out_dir: Path):
    print(f"Processing {img_path}")
    image = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size
    img_area = img_w * img_h

    masks = mask_generator.generate(np.asarray(image))

    # sort masks by area (largest first)
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    count = 0
    stem = img_path.stem

    for m in masks[:MAX_MASKS_PER_IMAGE]:
        area_ratio = m["area"] / img_area
        if area_ratio < MIN_AREA_RATIO:
            continue

        mask = m["segmentation"]
        bbox = get_expanded_bbox(mask, scale=1.2, img_w=img_w, img_h=img_h)
        if bbox is None:
            continue

        obj_img = extract_object(image, mask, bbox)

        out_path = out_dir / f"{stem}_{count}.png"
        obj_img.save(out_path)
        count += 1

    print(f"  -> saved {count} objects")


def main():
    sam_generator = load_sam()

    # iterate over category subfolders under ROOT_DIR
    for cat_dir in ROOT_DIR.iterdir():
        if not cat_dir.is_dir():
            continue

        # create /individual inside category
        out_dir = cat_dir / "individual"
        out_dir.mkdir(exist_ok=True)

        # process all images in this category folder
        for img_path in sorted(cat_dir.iterdir()):
            if img_path.is_dir():
                continue
            if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                continue
            process_image(sam_generator, img_path, out_dir)


if __name__ == "__main__":
    main()
