#!/usr/bin/env python3
"""
apply_masks_to_original_views.py

Your clarification:
- The masks in the ORIGINAL segmentation folder are ALREADY correct full-canvas masks
  (i.e., same pixel coordinate system as the original sketch views).
- So we do NOT do any bbox-based recovery / un-cropping.
- We simply:
  1) Read masks from: sketch/segmentation/0/view_{x}/*_mask.png
  2) Read original sketch image: sketch/views/view_{x}.png
  3) Copy-paste the mask to output (optionally binarize)
  4) Produce:
       - <stem>_mask.png     (copied full-canvas mask)
       - <stem>.png          (full-canvas cutout on white bg)
       - <stem>_overlay.png  (overlay visualization)
     and
       - all_components_bbox.json/png (bboxes computed from the mask)

Output directory (your latest):
  sketch/segmentation_original_mask/view_{x}/
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


# ---------------- config ----------------

BASE_PATH = Path("sketch")

# INPUT: masks are here (instance folder)
SEG_MASK_DIR = BASE_PATH / "segmentation" / "0"  # view_0/, view_1/, ...

# INPUT: original sketch views
ORIG_VIEWS_DIR = BASE_PATH / "views"             # view_0.png, view_1.png, ...

# OUTPUT: your requested folder
OUT_DIR = BASE_PATH / "segmentation_original_image"  # view_0/, view_1/, ...

# mask threshold for binarization (if masks are 0/255 already, this is fine)
MASK_THRESH = 127

# overlay alpha for visualization
OVERLAY_ALPHA = 110

# draw bbox outlines
BBOX_OUTLINE = (0, 160, 255)
BBOX_WIDTH = 3


# ---------------- helpers ----------------

def _is_view_folder(p: Path) -> bool:
    return p.is_dir() and re.match(r"view_\d+$", p.name) is not None


def _load_mask_full(mask_path: Path, W: int, H: int) -> Image.Image:
    """
    Load a full-canvas mask, ensure it matches (W,H), and binarize to 0/255.
    """
    m = Image.open(mask_path).convert("L")
    if m.size != (W, H):
        raise ValueError(
            f"Mask size {m.size} != original image size {(W,H)} for {mask_path}. "
            f"If your masks are correct, they must match exactly."
        )
    arr = np.array(m)
    arr_bin = (arr > MASK_THRESH).astype(np.uint8) * 255
    return Image.fromarray(arr_bin, mode="L")


def _mask_to_bbox(mask_bin: np.ndarray) -> Tuple[int, int, int, int]:
    """
    mask_bin: HxW uint8 with values 0 or 255
    Returns x0,y0,x1,y1 (x1,y1 exclusive)
    """
    ys, xs = np.where(mask_bin > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    return x0, y0, x1, y1


def apply_mask_outputs(
    orig_img: Image.Image,
    mask_full: Image.Image,
    out_view_dir: Path,
    stem: str,
    overlay_color=(0, 160, 255),
) -> Dict[str, Any]:
    """
    Saves:
      - <stem>_mask.png   (full canvas, binary)
      - <stem>.png        (full canvas cutout on white background)
      - <stem>_overlay.png (overlay on original)
    Returns bbox record.
    """
    out_view_dir.mkdir(parents=True, exist_ok=True)
    W, H = orig_img.size

    if mask_full.size != (W, H):
        raise ValueError(f"mask_full.size {mask_full.size} != orig size {(W,H)} for {stem}")

    # Save copied full mask
    mask_full_path = out_view_dir / f"{stem}_mask.png"
    mask_full.save(mask_full_path)

    # Full-canvas cutout (white background)
    comp = Image.new("RGB", (W, H), (255, 255, 255))
    comp.paste(orig_img.convert("RGB"), (0, 0), mask_full)
    comp_path = out_view_dir / f"{stem}.png"
    comp.save(comp_path)

    # Overlay visualization
    overlay = orig_img.copy().convert("RGBA")
    color_layer = Image.new("RGBA", (W, H), overlay_color + (0,))
    alpha = Image.fromarray(((np.array(mask_full) > 0).astype(np.uint8) * OVERLAY_ALPHA), mode="L")
    color_layer.putalpha(alpha)
    overlay.alpha_composite(color_layer)
    overlay_path = out_view_dir / f"{stem}_overlay.png"
    overlay.save(overlay_path)

    # Compute bbox from mask
    mask_bin = np.array(mask_full, dtype=np.uint8)
    x0, y0, x1, y1 = _mask_to_bbox(mask_bin)
    is_empty = (x1 <= x0) or (y1 <= y0)

    return {
        "label": stem,
        "box_xyxy": [float(x0), float(y0), float(x1), float(y1)],
        "box_xywh": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
        "mask_path": f"{stem}_mask.png",
        "overlay_path": f"{stem}_overlay.png",
        "image_path": f"{stem}.png",
        "is_empty": bool(is_empty),
    }


def process_view(view_dir: Path):
    """
    view_dir: sketch/segmentation/0/view_x
    Uses masks in that folder and overlays onto sketch/views/view_x.png
    Writes outputs to sketch/segmentation_original_mask/view_x/
    """
    view_name = view_dir.name
    orig_img_path = ORIG_VIEWS_DIR / f"{view_name}.png"
    if not orig_img_path.exists():
        print(f"[WARN] Missing original view image: {orig_img_path} (skip)")
        return

    orig_img = Image.open(orig_img_path).convert("RGB")
    W, H = orig_img.size

    out_view_dir = OUT_DIR / view_name
    out_view_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(view_dir.glob("*_mask.png"))
    if not mask_paths:
        print(f"[WARN] No *_mask.png found in {view_dir} (skip)")
        return

    bbox_records: List[Dict[str, Any]] = []

    for mp in mask_paths:
        # "<stem>_mask.png" -> stem
        stem = mp.stem
        if stem.endswith("_mask"):
            stem = stem[:-5]

        try:
            mask_full = _load_mask_full(mp, W=W, H=H)
        except Exception as e:
            print(f"[WARN] {view_name} {mp.name}: {e} (skip)")
            continue

        rec = apply_mask_outputs(
            orig_img=orig_img,
            mask_full=mask_full,
            out_view_dir=out_view_dir,
            stem=stem,
        )
        bbox_records.append(rec)

    # Save bbox visualization image
    bbox_img = orig_img.copy()
    draw = ImageDraw.Draw(bbox_img)
    for r in bbox_records:
        x0, y0, x1, y1 = r["box_xyxy"]
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0, y0, x1, y1], outline=BBOX_OUTLINE, width=BBOX_WIDTH)
        draw.text((x0, y0), r["label"], fill=BBOX_OUTLINE)
    bbox_img.save(out_view_dir / "all_components_bbox.png")

    # Save bbox json
    with open(out_view_dir / "all_components_bbox.json", "w") as f:
        json.dump(
            {
                "view": view_name,
                "image": str(orig_img_path),
                "image_size": {"width": int(W), "height": int(H)},
                "mask_source_dir": str(view_dir),
                "detections": bbox_records,
            },
            f,
            indent=2,
        )

    print(f"[OK] {view_name}: wrote {len(bbox_records)} masks+overlays to {out_view_dir}")


def main():
    if not SEG_MASK_DIR.exists():
        raise SystemExit(f"Missing input mask dir: {SEG_MASK_DIR}")
    if not ORIG_VIEWS_DIR.exists():
        raise SystemExit(f"Missing original views dir: {ORIG_VIEWS_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    view_dirs = sorted([p for p in SEG_MASK_DIR.iterdir() if _is_view_folder(p)])
    if not view_dirs:
        raise SystemExit(f"No view_* folders found under: {SEG_MASK_DIR}")

    for vd in view_dirs:
        process_view(vd)

    print("[DONE] Copied full masks and overlaid them onto original sketch views.")


if __name__ == "__main__":
    main()
