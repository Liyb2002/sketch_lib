#!/usr/bin/env python3
"""
generate_views_all.py — run Zero123++ multi-view generation
for every image in every object folder under ./sketches.

Folder structure (input):
    sketches/
        car/
            a.png
            b.png
        chair/
            foo.png
            bar.png

Outputs:
    sketches/car/views/a_grid.png
    sketches/car/views/a_view_0.png ... a_view_5.png
    sketches/car/views/b_grid.png
    ...
"""

from pathlib import Path

import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# ---------------------------------------------------------------------
# CONFIG: Zero123++ model + custom pipeline
# ---------------------------------------------------------------------

MODEL_ID = "sudo-ai/zero123plus-v1.2"
CUSTOM_PIPELINE = "sudo-ai/zero123plus-pipeline"  # or None

# Zero123++ usually outputs a 3x2 grid (6 views)
N_COLS = 3
N_ROWS = 2

IMG_SIZE = 320  # input size for conditioning image

# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SKETCHES_DIR = ROOT / "sketches"


def load_square_image(path: Path, size: int = IMG_SIZE) -> Image.Image:
    """Load image, pad to square on white background, resize to (size, size)."""
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = max(w, h)
    bg = Image.new("RGB", (side, side), "white")
    bg.paste(img, ((side - w) // 2, (side - h) // 2))
    return bg.resize((size, size), Image.BICUBIC)


def crop_grid_to_views(grid_img: Image.Image, n_cols: int, n_rows: int):
    """Split a grid image into n_cols * n_rows tiles (row-major order)."""
    w, h = grid_img.size
    tile_w = w // n_cols
    tile_h = h // n_rows
    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            box = (c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)
            tiles.append(grid_img.crop(box))
    return tiles


def find_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    # Only files directly under folder (ignore subfolders like 'views', 'individual', etc.)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main():
    if not SKETCHES_DIR.is_dir():
        raise FileNotFoundError(f"Base sketches folder not found: {SKETCHES_DIR}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Activate zero123pp_env (or similar) with GPU support.")

    # Load model once, reuse for everything
    print(f"[NVS] Loading model: {MODEL_ID}")
    pipe_kwargs = {
        "pretrained_model_name_or_path": MODEL_ID,
        "torch_dtype": torch.float16,
    }
    if CUSTOM_PIPELINE is not None:
        pipe_kwargs["custom_pipeline"] = CUSTOM_PIPELINE

    pipe = DiffusionPipeline.from_pretrained(**pipe_kwargs)

    # Optional: try EulerAncestral scheduler
    try:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    except Exception as e:
        print(f"[NVS] Could not switch scheduler to EulerAncestral: {e}")
        print("[NVS] Using default scheduler.")

    pipe.to("cuda")
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    # Iterate over all object folders
    object_folders = [p for p in SKETCHES_DIR.iterdir() if p.is_dir()]
    if not object_folders:
        print(f"[NVS] No object folders found under {SKETCHES_DIR}")
        return

    print(f"[NVS] Found {len(object_folders)} object folders.")

    for obj_dir in object_folders:
        print(f"\n[NVS] Object: {obj_dir.name}")

        images = find_images(obj_dir)
        if not images:
            print(f"  No images found in {obj_dir}, skipping.")
            continue

        # Create a shared 'views' folder for this object
        views_root = obj_dir / "views"
        views_root.mkdir(parents=True, exist_ok=True)

        print(f"  Found {len(images)} images.")

        for img_path in images:
            print(f"  [NVS] Processing: {img_path.name}")

            cond_img = load_square_image(img_path, size=IMG_SIZE)

            with torch.autocast("cuda"):
                out = pipe(cond_img, num_inference_steps=50)

            if hasattr(out, "images"):
                grid = out.images[0]
            elif isinstance(out, list):
                grid = out[0]
            else:
                raise RuntimeError("Unexpected pipeline output format from Zero123++.")

            stem = img_path.stem

            grid_path = views_root / f"{stem}_grid.png"
            grid.save(grid_path)
            print(f"    Saved grid: {grid_path}")

            views = crop_grid_to_views(grid, N_COLS, N_ROWS)
            print(f"    Cropping into {len(views)} views...")

            for i, v in enumerate(views):
                view_path = views_root / f"{stem}_view_{i}.png"
                v.save(view_path)
                print(f"      saved {view_path}")

    print("\n[NVS] All done.")


if __name__ == "__main__":
    main()
