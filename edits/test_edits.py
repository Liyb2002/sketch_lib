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
    Given:
        root/view_0.png
        root/view_0/{label_prefix}_0_mask.png

    1) Find the mask bbox.
    2) Make bbox taller in Y by `scale_y`, keeping the same bottom y.
    3) Crop the component patch from the base image using original bbox + mask.
    4) Stretch the patch vertically to fit the new taller bbox.
    5) Paste stretched patch back into the base image.

    Saves:
        root/view_0_<label_prefix>_stretched.png  (by default)
    """
    root = Path(root)

    # ---------------- paths ----------------
    base_img_path = root / f"view_{view_id}.png"
    view_folder = root / f"view_{view_id}"

    mask_path = next(view_folder.glob(f"{label_prefix}_*_mask.png"))

    # output
    if out_name is None:
        out_name = f"view_{view_id}_{label_prefix}_stretched.png"
    out_path = root / out_name

    # ---------------- load images ----------------
    base = Image.open(base_img_path).convert("RGBA")
    base_w, base_h = base.size

    mask = Image.open(mask_path).convert("L")  # single-channel mask
    mask_np = np.array(mask)

    # ---------------- original bbox from mask ----------------
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask appears to be empty.")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    orig_w = x_max - x_min + 1
    orig_h = y_max - y_min + 1

    # ---------------- new taller bbox ----------------
    new_h = int(orig_h * scale_y)

    # keep same bottom y (y_max), grow upwards
    new_y_min = y_max - new_h + 1
    new_y_max = y_max

    # clamp to image bounds if needed
    if new_y_min < 0:
        new_y_min = 0
        new_y_max = new_y_min + new_h - 1
    if new_y_max >= base_h:
        new_y_max = base_h - 1
        new_y_min = new_y_max - new_h + 1

    # x-span stays the same
    new_x_min, new_x_max = x_min, x_max

    # safety clamp for x as well
    new_x_min = max(0, new_x_min)
    new_x_max = min(base_w - 1, new_x_max)

    # recompute width/height from clamped bbox
    new_w = new_x_max - new_x_min + 1
    new_h = new_y_max - new_y_min + 1

    # ---------------- extract original component patch ----------------
    # crop from base image (RGBA)
    orig_bbox = (x_min, y_min, x_max + 1, y_max + 1)
    patch_rgb = base.crop(orig_bbox)

    # crop corresponding mask and use as alpha
    mask_crop = mask.crop(orig_bbox)
    patch_rgba = patch_rgb.convert("RGBA")
    patch_rgba.putalpha(mask_crop)

    # ---------------- resize patch into new bbox space ----------------
    resized_patch = patch_rgba.resize((new_w, new_h), Image.BILINEAR)

    # ---------------- paste back into base with alpha ----------------
    edited = base.copy()
    edited.alpha_composite(
        resized_patch,
        (new_x_min, new_y_min)
    )

    # ---------------- save ----------------
    edited.save(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    # Example: test_chair/0, backrest stretched 2x in height
    stretch_component_y(Path("test_chair/0"), view_id=0, label_prefix="backrest", scale_y=2.0)
