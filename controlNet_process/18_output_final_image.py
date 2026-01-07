#!/usr/bin/env python3
# 18_output_final_image.py
"""
Step 2: REMOVE pixels using silhouette masks (verification-passed version).

For each view:
1) Load original image:              sketch/views/view_{x}.png
2) Read bbox edits:                  sketch/final_results/view_{x}/bbox_edits.json
3) For each changed label:
   - Load overlay mask image:        sketch/segmentation_original_image/view_{x}/{label}_mask.png
   - Convert overlay mask -> silhouette
   - Resize silhouette to ORIGINAL bbox size
   - Place silhouette at ORIGINAL bbox location
4) Remove pixels under silhouette (paint PURE WHITE)

Output:
  sketch/final_results/view_{x}/mask_removed.png
"""

import os
import json
import argparse
from typing import Any, List, Tuple

import numpy as np


# ------------------------ IO ------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ------------------------ Image utils ------------------------

def load_image_any(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA") if "A" in img.getbands() else img.convert("RGB")
    return np.array(img)

def to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        return img[..., :3]
    return img

def save_image_rgb(path: str, img_rgb: np.ndarray) -> None:
    from PIL import Image
    ensure_dir(os.path.dirname(path))
    Image.fromarray(img_rgb.astype(np.uint8), mode="RGB").save(path)

def resize_nearest(mask_u8: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray(mask_u8.astype(np.uint8), mode="L")
    im2 = im.resize((int(new_w), int(new_h)), resample=Image.NEAREST)
    return np.array(im2)


# ------------------------ BBox utils ------------------------

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def xyxy_float_to_int_box(box_xyxy: List[float], W: int, H: int) -> Tuple[int,int,int,int]:
    x0, y0, x1, y1 = [float(x) for x in box_xyxy]

    xi0 = clamp(int(np.floor(x0)), 0, W)
    yi0 = clamp(int(np.floor(y0)), 0, H)
    xi1 = clamp(int(np.ceil(x1)), 0, W)
    yi1 = clamp(int(np.ceil(y1)), 0, H)

    if xi1 <= xi0:
        xi1 = min(W, xi0 + 1)
    if yi1 <= yi0:
        yi1 = min(H, yi0 + 1)

    return xi0, yi0, xi1, yi1


# ------------------------ Silhouette extraction ------------------------

def overlay_mask_to_silhouette(mask_img: np.ndarray) -> np.ndarray:
    """
    Convert overlay mask image to silhouette (uint8 0 or 255).

    Rules:
    - RGBA: alpha > 0
    - RGB:  non-white pixels
    - L:    >0
    """
    if mask_img.ndim == 2:
        fg = mask_img > 0
        return fg.astype(np.uint8) * 255

    if mask_img.shape[-1] == 4:
        alpha = mask_img[..., 3]
        fg = alpha > 0
        return fg.astype(np.uint8) * 255

    rgb = mask_img[..., :3]
    fg = np.any(rgb < 250, axis=-1)
    return fg.astype(np.uint8) * 255


# ------------------------ Main ------------------------

def main(
    num_views: int,
    views_dir: str,
    seg_orig_root: str,
    edits_root: str,
    out_root: str,
):
    for vid in range(num_views):
        view = f"view_{vid}"

        img_path = os.path.join(views_dir, f"{view}.png")
        edits_path = os.path.join(edits_root, view, "bbox_edits.json")
        seg_view_dir = os.path.join(seg_orig_root, view)

        if not os.path.isfile(img_path):
            raise FileNotFoundError(img_path)
        if not os.path.isfile(edits_path):
            raise FileNotFoundError(edits_path)
        if not os.path.isdir(seg_view_dir):
            raise FileNotFoundError(seg_view_dir)

        img_rgb = to_rgb(load_image_any(img_path)).copy()
        H, W = img_rgb.shape[:2]

        edits = load_json(edits_path)
        changed = edits.get("changed_labels", [])

        # Full canvas removal mask
        removal = np.zeros((H, W), dtype=np.uint8)

        for item in changed:
            label = str(item.get("label", ""))
            orig_box = item.get("original_box_xyxy", None)
            if not label or not (isinstance(orig_box, list) and len(orig_box) == 4):
                continue

            mask_path = os.path.join(seg_view_dir, f"{label}_mask.png")
            if not os.path.isfile(mask_path):
                print(f"[skip] missing mask: {mask_path}")
                continue

            mask_img = load_image_any(mask_path)
            sil = overlay_mask_to_silhouette(mask_img)

            x0, y0, x1, y1 = xyxy_float_to_int_box(orig_box, W, H)
            bw, bh = x1 - x0, y1 - y0
            if bw <= 0 or bh <= 0:
                continue

            sil_rs = resize_nearest(sil, bw, bh)
            removal[y0:y1, x0:x1] = np.maximum(
                removal[y0:y1, x0:x1],
                sil_rs
            )

        # REMOVE = paint white
        img_rgb[removal > 0] = np.array([255, 255, 255], dtype=np.uint8)

        out_dir = os.path.join(out_root, view)
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, "mask_removed.png")
        save_image_rgb(out_path, img_rgb)

        print(f"[ok] {view} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--views_dir", default="sketch/views")
    parser.add_argument("--seg_orig_root", default="sketch/segmentation_original_image")
    parser.add_argument("--edits_root", default="sketch/final_results")
    parser.add_argument("--out_root", default="sketch/final_results")
    args = parser.parse_args()

    main(
        num_views=args.num_views,
        views_dir=args.views_dir,
        seg_orig_root=args.seg_orig_root,
        edits_root=args.edits_root,
        out_root=args.out_root,
    )
