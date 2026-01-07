#!/usr/bin/env python3
"""
apply_masks_to_original_views.py

You already segmented on "realistic" images and saved per-view masks under:
  sketch/segmentation/view_{x}/<label>_<i>_mask.png

Now this script:
1) Creates: sketch/segmentation_original_image/view_{x}/
2) For each view_x, for each mask, applies it onto the ORIGINAL image:
     sketch/views/view_{x}.png
   and saves a masked crop (white background) with the SAME stem:
     <stem>.png   (e.g., wheel_0.png)
   plus an optional overlay:
     <stem>_overlay.png
3) Also writes a bbox json for convenience:
     all_components_bbox.json
   (copied/derived from the realistic-view json, but paths updated)

Assumptions:
- view folders in sketch/segmentation are named like "view_0", "view_1", ...
- original images are at sketch/views/view_0.png, ..., view_5.png (or any subset)
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


# ---------------- config ----------------

BASE_PATH = Path("sketch")

# masks + bbox json produced by your existing SAM3 script (on realistic images)
SEG_REALISTIC_DIR = BASE_PATH / "segmentation"  # contains view_0/, view_1/, ...

# original images you want to cut from
ORIG_VIEWS_DIR = BASE_PATH / "views"            # contains view_0.png, view_1.png, ...

# outputs (masked crops) on original images
SEG_ORIG_OUT_DIR = BASE_PATH / "segmentation_original_image"

# overlay alpha when compositing colored overlay on original (0..255)
OVERLAY_ALPHA = 110

# If your masks are L-mode 0/255, this threshold is fine.
MASK_THRESH = 127

# bbox expansion factor (optional). Set to 1.0 to keep exact bbox.
BBOX_SCALE = 1.0


# ---------------- helpers ----------------

def _parse_view_index(view_name: str) -> int:
    """
    view_name: "view_0" -> 0
    """
    m = re.match(r"view_(\d+)$", view_name)
    return int(m.group(1)) if m else -1


def _load_mask(mask_path: Path) -> np.ndarray:
    """
    Returns a boolean mask HxW.
    """
    m = Image.open(mask_path).convert("L")
    arr = np.array(m)
    return arr > MASK_THRESH


def _mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Returns tight bbox (x0,y0,x1,y1) from boolean mask.
    x1,y1 are exclusive (PIL crop style).
    """
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return int(x0), int(y0), int(x1), int(y1)


def _scale_bbox(x0: int, y0: int, x1: int, y1: int, W: int, H: int, scale: float) -> Tuple[int, int, int, int]:
    if scale <= 1.0:
        return x0, y0, x1, y1
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = (x1 - x0) * scale
    bh = (y1 - y0) * scale
    nx0 = max(0, int(round(cx - bw / 2)))
    ny0 = max(0, int(round(cy - bh / 2)))
    nx1 = min(W, int(round(cx + bw / 2)))
    ny1 = min(H, int(round(cy + bh / 2)))
    return nx0, ny0, nx1, ny1


def apply_one_mask_to_original(
    orig_img: Image.Image,
    mask_bool: np.ndarray,
    out_dir: Path,
    stem: str,
    overlay_color=(0, 160, 255),
) -> Dict[str, Any]:
    """
    Saves:
      - <stem>.png             masked crop on white background
      - <stem>_overlay.png     overlay on full original image (optional)
    Returns bbox record dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = orig_img.size

    if mask_bool.shape != (H, W):
        raise ValueError(f"Mask size {mask_bool.shape} does not match image size {(H, W)} for {stem}")

    x0, y0, x1, y1 = _mask_to_bbox(mask_bool)
    if x1 <= x0 or y1 <= y0:
        # Empty mask
        return {
            "label": stem,
            "box_xyxy": [0.0, 0.0, 0.0, 0.0],
            "box_xywh": [0.0, 0.0, 0.0, 0.0],
            "mask_path": f"{stem}_mask.png",
            "overlay_path": f"{stem}_overlay.png",
            "crop_path": f"{stem}.png",
            "is_empty": True,
        }

    x0, y0, x1, y1 = _scale_bbox(x0, y0, x1, y1, W, H, BBOX_SCALE)

    # Crop image and mask
    crop_img = orig_img.crop((x0, y0, x1, y1)).convert("RGB")
    crop_mask = mask_bool[y0:y1, x0:x1]

    # Save masked crop on white bg
    comp = Image.new("RGB", crop_img.size, (255, 255, 255))
    alpha = Image.fromarray((crop_mask.astype(np.uint8) * 255), mode="L")
    comp.paste(crop_img, (0, 0), alpha)
    comp.save(out_dir / f"{stem}.png")

    # Also save the mask (cropped) for convenience
    alpha.save(out_dir / f"{stem}_mask.png")

    # Save overlay on full original (optional but handy)
    overlay = orig_img.copy().convert("RGBA")
    color_layer = Image.new("RGBA", overlay.size, overlay_color + (0,))
    full_alpha = Image.fromarray((mask_bool.astype(np.uint8) * OVERLAY_ALPHA), mode="L")
    color_layer.putalpha(full_alpha)
    overlay.alpha_composite(color_layer)
    overlay.save(out_dir / f"{stem}_overlay.png")

    return {
        "label": stem,
        "box_xyxy": [float(x0), float(y0), float(x1), float(y1)],
        "box_xywh": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
        "mask_path": f"{stem}_mask.png",
        "overlay_path": f"{stem}_overlay.png",
        "crop_path": f"{stem}.png",
        "is_empty": False,
    }


def process_view(view_dir: Path):
    """
    view_dir is like sketch/segmentation/view_0
    Uses masks in that folder and applies them to sketch/views/view_0.png
    """
    view_name = view_dir.name  # "view_0"
    idx = _parse_view_index(view_name)
    if idx < 0:
        return

    orig_img_path = ORIG_VIEWS_DIR / f"{view_name}.png"
    if not orig_img_path.exists():
        print(f"[WARN] Missing original view image: {orig_img_path} (skip)")
        return

    out_dir = SEG_ORIG_OUT_DIR / view_name
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_img = Image.open(orig_img_path).convert("RGB")
    W, H = orig_img.size

    # Gather mask files from realistic segmentation
    mask_paths = sorted(view_dir.glob("*_mask.png"))
    if not mask_paths:
        print(f"[WARN] No masks found in {view_dir} (skip)")
        return

    bbox_records: List[Dict[str, Any]] = []

    for mp in mask_paths:
        # mp name: "<stem>_mask.png" where stem like "wheel_0"
        stem = mp.stem
        if stem.endswith("_mask"):
            stem = stem[:-5]

        mask_bool = _load_mask(mp)

        rec = apply_one_mask_to_original(
            orig_img=orig_img,
            mask_bool=mask_bool,
            out_dir=out_dir,
            stem=stem,
        )
        bbox_records.append(rec)

    # Save a bbox visualization (optional)
    bbox_img = orig_img.copy()
    draw = ImageDraw.Draw(bbox_img)
    for r in bbox_records:
        x0, y0, x1, y1 = r["box_xyxy"]
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0, y0, x1, y1], outline=(0, 160, 255), width=3)
        draw.text((x0, y0), r["label"], fill=(0, 160, 255))
    bbox_img.save(out_dir / "all_components_bbox.png")

    # Save bbox json
    with open(out_dir / "all_components_bbox.json", "w") as f:
        json.dump(
            {
                "view": view_name,
                "image": str(orig_img_path),
                "image_size": {"width": int(W), "height": int(H)},
                "mask_source_dir": str(view_dir),
                "bbox_scale": float(BBOX_SCALE),
                "detections": bbox_records,
            },
            f,
            indent=2,
        )

    print(f"[OK] {view_name}: wrote {len(bbox_records)} components to {out_dir}")


def main():
    if not SEG_REALISTIC_DIR.exists():
        raise SystemExit(f"Missing realistic segmentation dir: {SEG_REALISTIC_DIR}")
    if not ORIG_VIEWS_DIR.exists():
        raise SystemExit(f"Missing original views dir: {ORIG_VIEWS_DIR}")

    SEG_ORIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

    view_dirs = sorted([p for p in SEG_REALISTIC_DIR.iterdir() if p.is_dir() and re.match(r"view_\d+$", p.name)])
    if not view_dirs:
        raise SystemExit(f"No view_* folders found under: {SEG_REALISTIC_DIR}")

    for vd in view_dirs:
        process_view(vd)

    print("[DONE] Applied realistic masks onto original images.")


if __name__ == "__main__":
    main()
