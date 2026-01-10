#!/usr/bin/env python3
# 18_output_final_image.py
"""
Step 3: REMOVE original pixels by silhouette, then INSERT resized component into NEW bbox.

Per view:
1) Load original image: sketch/views/view_{x}.png
2) Load bbox edits:     sketch/final_results/view_{x}/bbox_edits.json
3) For each changed label:
   A) Build a full-canvas silhouette removal mask by:
      - load overlay mask: sketch/segmentation_original_image/view_{x}/{label}_mask.png
      - silhouette = alpha>0 OR non-white
      - resize silhouette to ORIGINAL bbox size
      - paste silhouette into canvas at ORIGINAL bbox location
   B) (Improved) Dilate removal mask by N pixels to remove halos
   C) Remove: paint white where removal mask is 1
   D) Save intermediate: removed_original.png
   E) Insert:
      - load component: sketch/segmentation_original_image/view_{x}/{label}.png
      - resize component to NEW bbox size
      - paste into NEW bbox location
        * prefer using the resized silhouette as alpha if component has no alpha
        * if component has alpha, alpha composite directly

Output:
  sketch/final_results/view_{x}/removed_original.png
  sketch/final_results/view_{x}/final_image.png
"""

import os
import json
import argparse
from typing import Any, List, Tuple, Dict

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

def save_mask_l(path: str, mask_u8: np.ndarray) -> None:
    """Optional: save a grayscale mask image (0..255)."""
    from PIL import Image
    ensure_dir(os.path.dirname(path))
    Image.fromarray(mask_u8.astype(np.uint8), mode="L").save(path)

def resize_bicubic(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray(img)
    pil2 = pil.resize((int(new_w), int(new_h)), resample=Image.BICUBIC)
    return np.array(pil2)

def resize_nearest(mask_u8: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray(mask_u8.astype(np.uint8), mode="L")
    im2 = im.resize((int(new_w), int(new_h)), resample=Image.NEAREST)
    return np.array(im2)

def dilate_mask(mask_u8: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Binary dilation for a uint8 mask (0..255). Returns uint8 (0 or 255).
    Uses PIL MaxFilter (fast, no extra deps).
    """
    if radius_px <= 0:
        return mask_u8
    from PIL import Image, ImageFilter
    k = int(2 * radius_px + 1)  # must be odd
    im = Image.fromarray(((mask_u8 > 0).astype(np.uint8) * 255), mode="L")
    im2 = im.filter(ImageFilter.MaxFilter(size=k))
    return np.array(im2, dtype=np.uint8)

def alpha_composite(dst_rgb: np.ndarray, src_rgba: np.ndarray, x0: int, y0: int) -> None:
    """
    In-place alpha composite src_rgba onto dst_rgb at top-left (x0,y0).
    dst_rgb: RGB uint8
    src_rgba: RGBA uint8
    """
    H, W = dst_rgb.shape[:2]
    sh, sw = src_rgba.shape[:2]
    x1 = min(W, x0 + sw)
    y1 = min(H, y0 + sh)
    if x1 <= x0 or y1 <= y0:
        return

    src = src_rgba[:(y1 - y0), :(x1 - x0), :].astype(np.float32)
    dst = dst_rgb[y0:y1, x0:x1, :].astype(np.float32)

    rgb = src[..., :3]
    a = src[..., 3:4] / 255.0

    out = dst * (1.0 - a) + rgb * a
    dst_rgb[y0:y1, x0:x1, :] = np.clip(out, 0, 255).astype(np.uint8)

def paste_with_alpha(dst_rgb: np.ndarray, src_rgb: np.ndarray, alpha_u8: np.ndarray, x0: int, y0: int) -> None:
    """
    In-place paste src_rgb using alpha_u8 (0..255) at (x0,y0).
    """
    H, W = dst_rgb.shape[:2]
    sh, sw = src_rgb.shape[:2]
    x1 = min(W, x0 + sw)
    y1 = min(H, y0 + sh)
    if x1 <= x0 or y1 <= y0:
        return

    src_crop = src_rgb[:(y1 - y0), :(x1 - x0), :].astype(np.float32)
    a_crop = alpha_u8[:(y1 - y0), :(x1 - x0)].astype(np.float32) / 255.0
    a_crop = a_crop[..., None]

    dst_crop = dst_rgb[y0:y1, x0:x1, :].astype(np.float32)
    out = dst_crop * (1.0 - a_crop) + src_crop * a_crop
    dst_rgb[y0:y1, x0:x1, :] = np.clip(out, 0, 255).astype(np.uint8)


# ------------------------ BBox utils ------------------------

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def xyxy_float_to_int_box(box_xyxy: List[float], W: int, H: int) -> Tuple[int, int, int, int]:
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
    - RGBA: alpha > 0
    - RGB:  non-white pixels
    - L:    >0
    """
    if mask_img.ndim == 2:
        return ((mask_img > 0).astype(np.uint8) * 255)

    if mask_img.shape[-1] == 4:
        alpha = mask_img[..., 3]
        return ((alpha > 0).astype(np.uint8) * 255)

    rgb = mask_img[..., :3]
    fg = np.any(rgb < 250, axis=-1)
    return (fg.astype(np.uint8) * 255)


# ------------------------ Main ------------------------

def main(
    num_views: int,
    views_dir: str,
    seg_orig_root: str,
    edits_root: str,
    out_root: str,
    removal_dilate_px: int = 5,
    save_removal_mask: bool = False,
):
    for vid in range(num_views):
        view = f"view_{vid}"

        img_path = os.path.join(views_dir, f"{view}.png")
        edits_path = os.path.join(edits_root, view, "bbox_edits.json")
        seg_view_dir = os.path.join(seg_orig_root, view)

        # NEW: if any view input is missing, skip instead of crashing
        if not os.path.isfile(img_path):
            print(f"[skip] missing view image: {img_path}")
            continue
        if not os.path.isfile(edits_path):
            print(f"[skip] missing bbox edits: {edits_path}")
            continue
        if not os.path.isdir(seg_view_dir):
            print(f"[skip] missing segmentation dir: {seg_view_dir}")
            continue

        img_rgb = to_rgb(load_image_any(img_path)).copy()
        H, W = img_rgb.shape[:2]

        edits = load_json(edits_path)
        changed = edits.get("changed_labels", [])
        if not isinstance(changed, list):
            print(f"[skip] invalid schema: {edits_path} changed_labels must be list")
            continue

        out_dir = os.path.join(out_root, view)
        ensure_dir(out_dir)

        # ---------- PASS 1: build full removal mask + remove ----------
        removal = np.zeros((H, W), dtype=np.uint8)

        # Store per-label silhouettes resized for NEW bbox (used for alpha if needed)
        new_alpha_by_label: Dict[str, np.ndarray] = {}

        for item in changed:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", ""))
            orig_box = item.get("original_box_xyxy", None)
            new_box = item.get("new_box_xyxy", None)
            if (
                not label
                or not (isinstance(orig_box, list) and len(orig_box) == 4)
                or not (isinstance(new_box, list) and len(new_box) == 4)
            ):
                continue

            mask_path = os.path.join(seg_view_dir, f"{label}_mask.png")
            if not os.path.isfile(mask_path):
                print(f"[skip] missing overlay mask: {mask_path}")
                continue

            sil = overlay_mask_to_silhouette(load_image_any(mask_path))

            # place for removal at ORIGINAL bbox
            ox0, oy0, ox1, oy1 = xyxy_float_to_int_box(orig_box, W, H)
            bw, bh = ox1 - ox0, oy1 - oy0
            sil_orig = resize_nearest(sil, bw, bh)
            removal[oy0:oy1, ox0:ox1] = np.maximum(removal[oy0:oy1, ox0:ox1], sil_orig)

            # prepare alpha for insertion at NEW bbox
            nx0, ny0, nx1, ny1 = xyxy_float_to_int_box(new_box, W, H)
            nw, nh = nx1 - nx0, ny1 - ny0
            sil_new = resize_nearest(sil, nw, nh)
            new_alpha_by_label[label] = sil_new

        # NEW: dilate removal mask to remove halo / outer layer
        removal = dilate_mask(removal, radius_px=removal_dilate_px)

        # Optional: save the removal mask itself
        if save_removal_mask:
            save_mask_l(os.path.join(out_dir, "removal_mask.png"), removal)

        # remove
        img_rgb[removal > 0] = np.array([255, 255, 255], dtype=np.uint8)

        # NEW: save intermediate after removal, before insertion
        removed_path = os.path.join(out_dir, "removed_original.png")
        save_image_rgb(removed_path, img_rgb)
        print(f"[ok] {view} -> {removed_path} (after removal)")

        # ---------- PASS 2: insert resized components ----------
        for item in changed:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", ""))
            new_box = item.get("new_box_xyxy", None)
            if not label or not (isinstance(new_box, list) and len(new_box) == 4):
                continue

            comp_path = os.path.join(seg_view_dir, f"{label}.png")
            if not os.path.isfile(comp_path):
                print(f"[skip] missing component png: {comp_path}")
                continue

            nx0, ny0, nx1, ny1 = xyxy_float_to_int_box(new_box, W, H)
            nw, nh = nx1 - nx0, ny1 - ny0
            if nw <= 0 or nh <= 0:
                continue

            comp = load_image_any(comp_path)  # RGB/RGBA/L
            comp_rs = resize_bicubic(comp, nw, nh)

            if comp_rs.ndim == 3 and comp_rs.shape[-1] == 4:
                # component has alpha -> best
                alpha_composite(img_rgb, comp_rs, nx0, ny0)
            else:
                # no alpha -> use silhouette alpha (best we can do)
                comp_rgb = to_rgb(comp_rs)
                alpha_u8 = new_alpha_by_label.get(label, None)
                if alpha_u8 is None:
                    # fallback: overwrite
                    x1 = min(W, nx0 + nw)
                    y1 = min(H, ny0 + nh)
                    img_rgb[ny0:y1, nx0:x1, :] = comp_rgb[:(y1 - ny0), :(x1 - nx0), :]
                else:
                    paste_with_alpha(img_rgb, comp_rgb, alpha_u8, nx0, ny0)

        out_path = os.path.join(out_dir, "final_image.png")
        save_image_rgb(out_path, img_rgb)
        print(f"[ok] {view} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--views_dir", default="sketch/views")
    parser.add_argument("--seg_orig_root", default="sketch/segmentation_original_image")
    parser.add_argument("--edits_root", default="sketch/final_results")
    parser.add_argument("--out_root", default="sketch/final_results")

    # NEW options
    parser.add_argument("--removal_dilate_px", type=int, default=5,
                        help="Dilation radius in pixels for removal mask (removes halo).")
    parser.add_argument("--save_removal_mask", action="store_true",
                        help="Also save removal_mask.png per view (debug).")

    args = parser.parse_args()

    main(
        num_views=args.num_views,
        views_dir=args.views_dir,
        seg_orig_root=args.seg_orig_root,
        edits_root=args.edits_root,
        out_root=args.out_root,
        removal_dilate_px=args.removal_dilate_px,
        save_removal_mask=args.save_removal_mask,
    )
