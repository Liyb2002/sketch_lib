#!/usr/bin/env python3
# 18_output_final_image.py
"""
Step 1 (verification only) — silhouette overlay in pure red.

You said the mask file is an "overlay mask" (typically a cropped mask/image),
so we:
1) Load original view image:    sketch/views/view_{x}.png
2) Read bbox_edits.json:        sketch/final_results/view_{x}/bbox_edits.json
3) For each changed label:
   - Load overlay mask image:   sketch/segmentation_original_image/view_{x}/{label}_mask.png
     (this can be cropped or full-size; can be RGB/RGBA/L)
   - Convert it to a silhouette (binary mask) by thresholding non-white (or alpha>0)
   - Place this silhouette onto the full image canvas at the ORIGINAL bbox location
     using original_box_xyxy from bbox_edits.json
   - Paint silhouette pixels as pure red (255,0,0) on the original image

Output:
  sketch/final_results/view_{x}/mask_remove_overlay.png

Nothing else. No whitening, no resizing, no pasting components.
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
    """
    Convert float bbox to int pixel bbox:
      left/top = floor, right/bot = ceil, then clamp.
    """
    x0, y0, x1, y1 = [float(x) for x in box_xyxy]
    xi0 = int(np.floor(x0))
    yi0 = int(np.floor(y0))
    xi1 = int(np.ceil(x1))
    yi1 = int(np.ceil(y1))

    xi0 = clamp(xi0, 0, W)
    xi1 = clamp(xi1, 0, W)
    yi0 = clamp(yi0, 0, H)
    yi1 = clamp(yi1, 0, H)

    if xi1 <= xi0:
        xi1 = min(W, xi0 + 1)
    if yi1 <= yi0:
        yi1 = min(H, yi0 + 1)

    return xi0, yi0, xi1, yi1


# ------------------------ Silhouette extraction ------------------------

def overlay_mask_to_silhouette(mask_img: np.ndarray) -> np.ndarray:
    """
    Convert an "overlay mask" image to a binary silhouette (uint8 0/255).

    Heuristics:
    - If RGBA: use alpha > 0 as foreground (preferred).
    - Else: treat "non-white" pixels as foreground:
        any(channel < 250) -> foreground
      (works for typical white background crops)
    """
    if mask_img.ndim == 2:
        # grayscale
        fg = mask_img > 0
        return (fg.astype(np.uint8) * 255)

    if mask_img.shape[-1] == 4:
        alpha = mask_img[..., 3]
        fg = alpha > 0
        return (fg.astype(np.uint8) * 255)

    rgb = mask_img[..., :3].astype(np.uint8)
    fg = np.any(rgb < 250, axis=-1)  # non-white
    return (fg.astype(np.uint8) * 255)


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
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Missing original image: {img_path}")

        edits_path = os.path.join(edits_root, view, "bbox_edits.json")
        if not os.path.isfile(edits_path):
            raise FileNotFoundError(f"Missing bbox edits: {edits_path}")

        seg_orig_view_dir = os.path.join(seg_orig_root, view)
        if not os.path.isdir(seg_orig_view_dir):
            raise FileNotFoundError(f"Missing folder: {seg_orig_view_dir}")

        img_rgb = to_rgb(load_image_any(img_path)).copy()
        H, W = img_rgb.shape[:2]

        edits = load_json(edits_path)
        changed = edits.get("changed_labels", [])
        if not isinstance(changed, list):
            raise ValueError(f"Invalid schema: {edits_path} changed_labels must be list")

        # Full-canvas union silhouette
        union = np.zeros((H, W), dtype=np.uint8)

        for item in changed:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", ""))
            orig_box = item.get("original_box_xyxy", None)
            if not label or not (isinstance(orig_box, list) and len(orig_box) == 4):
                continue

            mask_path = os.path.join(seg_orig_view_dir, f"{label}_mask.png")
            if not os.path.isfile(mask_path):
                print(f"[skip] missing overlay mask: {mask_path}")
                continue

            mask_img = load_image_any(mask_path)
            sil = overlay_mask_to_silhouette(mask_img)  # (h,w) uint8 0/255

            # Place silhouette at ORIGINAL bbox location.
            # We assume the overlay mask corresponds to that component crop, so it should be resized to bbox size.
            x0, y0, x1, y1 = xyxy_float_to_int_box(orig_box, W, H)
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)

            sil_rs = resize_nearest(sil, bw, bh)

            # Paste into union
            union[y0:y1, x0:x1] = np.maximum(union[y0:y1, x0:x1], sil_rs)

        # Paint silhouette as pure red on the image
        m = union > 0
        img_rgb[m] = np.array([255, 0, 0], dtype=np.uint8)

        out_dir = os.path.join(out_root, view)
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, "mask_remove_overlay.png")
        save_image_rgb(out_path, img_rgb)

        print(f"[ok] {view} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--views_dir", default="sketch/views", help="Contains view_{x}.png")
    parser.add_argument(
        "--seg_orig_root",
        default="sketch/segmentation_original_image",
        help="Contains view_{x}/{label}_mask.png overlay masks",
    )
    parser.add_argument(
        "--edits_root",
        default="sketch/final_results",
        help="Contains view_{x}/bbox_edits.json",
    )
    parser.add_argument(
        "--out_root",
        default="sketch/final_results",
        help="Output root; writes to view_{x}/mask_remove_overlay.png",
    )
    args = parser.parse_args()

    main(
        num_views=args.num_views,
        views_dir=args.views_dir,
        seg_orig_root=args.seg_orig_root,
        edits_root=args.edits_root,
        out_root=args.out_root,
    )
