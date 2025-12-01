#!/usr/bin/env python3
from pathlib import Path
from PIL import Image
import numpy as np


def stretch_component_y(
    root: Path,
    view_id: int = 0,
    label_prefix: str = "backrest",
    scale_y: float = 2.0,
    out_name: str | None = None,
):
    """
    Stretch a segmented component upward while keeping the bottom fixed.

    Inputs (for a given root):
        root/view_{view_id}.png
        root/view_{view_id}/{label_prefix}_*_mask.png

    Outputs in `root`:
        view_{view_id}_{label_prefix}_stretched.png       # edited RGB(A) image
        view_{view_id}_{label_prefix}_mask_before.png     # full-size original mask
        view_{view_id}_{label_prefix}_mask_after.png      # full-size stretched mask
    """
    root = Path(root)

    # ---------------- paths ----------------
    base_img_path = root / f"view_{view_id}.png"
    view_folder = root / f"view_{view_id}"

    mask_path = next(view_folder.glob(f"{label_prefix}_*_mask.png"))

    if out_name is None:
        out_name = f"view_{view_id}_{label_prefix}_stretched.png"
    out_path = root / out_name

    mask_before_path = root / f"view_{view_id}_{label_prefix}_mask_before.png"
    mask_after_path = root / f"view_{view_id}_{label_prefix}_mask_after.png"

    # ---------------- load images ----------------
    base = Image.open(base_img_path).convert("RGBA")
    base_w, base_h = base.size

    # original full-size mask (before)
    mask_before = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_before)

    # save original full-size mask directly
    mask_before.save(mask_before_path)

    # ---------------- original bbox from mask ----------------
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask appears to be empty.")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    orig_w = x_max - x_min + 1
    orig_h = y_max - y_min + 1

    # ---------------- new taller bbox ----------------
    # keep bottom fixed at y_max, grow upwards
    new_h = int(orig_h * scale_y)

    new_y_max = y_max
    new_y_min = new_y_max - new_h + 1

    # clamp to top edge without moving bottom
    if new_y_min < 0:
        new_y_min = 0
        new_h = new_y_max - new_y_min + 1

    # x-span stays the same, with safety clamp
    new_x_min, new_x_max = x_min, x_max
    new_x_min = max(0, new_x_min)
    new_x_max = min(base_w - 1, new_x_max)

    new_w = new_x_max - new_x_min + 1
    new_h = new_y_max - new_y_min + 1

    # ---------------- extract original component patch ----------------
    orig_bbox = (x_min, y_min, x_max + 1, y_max + 1)

    patch_rgb = base.crop(orig_bbox)
    mask_crop = mask_before.crop(orig_bbox)

    patch_rgba = patch_rgb.convert("RGBA")
    patch_rgba.putalpha(mask_crop)

    # ---------------- resize patch & mask ----------------
    resized_patch = patch_rgba.resize((new_w, new_h), Image.BILINEAR)
    resized_mask = mask_crop.resize((new_w, new_h), Image.NEAREST)

    # ---------------- paste into edited image ----------------
    edited = base.copy()
    edited.alpha_composite(resized_patch, (new_x_min, new_y_min))

    # ---------------- build full-size "after" mask ----------------
    mask_after_full = Image.new("L", (base_w, base_h), 0)
    mask_after_full.paste(resized_mask, (new_x_min, new_y_min))

    # ---------------- save outputs ----------------
    edited.save(out_path)
    mask_after_full.save(mask_after_path)

    print("Saved:", out_path)
    print("Saved:", mask_before_path)
    print("Saved:", mask_after_path)


if __name__ == "__main__":
    # Example: test_chair/0, backrest stretched 2x in height
    stretch_component_y(Path("test_chair/0"),
                        view_id=0,
                        label_prefix="backrest",
                        scale_y=2.0)
