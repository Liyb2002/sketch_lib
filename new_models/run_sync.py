#!/usr/bin/env python3
from pathlib import Path
import subprocess
from shutil import copyfile
from PIL import Image

# ---------------------------------------------------------------------
# Hard-coded paths and settings
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "SyncDreamer"

CKPT = REPO / "ckpt" / "syncdreamer-pretrain.ckpt"

INPUT_RGB = ROOT / "0.png"
INPUT_RGBA = ROOT / "0_rgba.png"

OUTPUT_DIR = REPO / "output" / "obj0"
RESULT_DIR = ROOT / "test_result"

# Number of samples and per-strip layout
NUM_SAMPLES = 2      # expect 0.png and 1.png
VIEWS_PER_STRIP = 16 # 1 x 16 layout

# How many views you ultimately want
MAX_VIEWS = 20

# Upscaled resolution for each view (change if you want)
UPSCALE_SIZE = (512, 512)


def ensure_rgba():
    """Make sure we have 0_rgba.png for SyncDreamer."""
    if INPUT_RGBA.exists():
        print(f"[info] Using existing {INPUT_RGBA}")
        return

    if not INPUT_RGB.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT_RGB}")

    img = Image.open(INPUT_RGB).convert("RGBA")
    img.save(INPUT_RGBA)
    print(f"[info] Saved RGBA input as {INPUT_RGBA}")


def run_syncdreamer():
    """Run SyncDreamer generate.py with sample_num=NUM_SAMPLES."""
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint missing: {CKPT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "generate.py",
        "--ckpt", str(CKPT),
        "--input", str(INPUT_RGBA),
        "--output", str(OUTPUT_DIR),
        "--sample_num", str(NUM_SAMPLES),
        "--cfg_scale", "2.0",
        "--elevation", "20",
        "--crop_size", "200",
    ]

    print("[info] Running SyncDreamer:")
    print("       ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO))
    print("[info] SyncDreamer done.")


def copy_full_strips():
    """Copy full strip images into test_result as grid_*.png."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for sid in range(NUM_SAMPLES):
        src = OUTPUT_DIR / f"{sid}.png"
        dst = RESULT_DIR / f"grid_{sid}.png"
        if src.exists():
            copyfile(src, dst)
            print(f"[info] Copied {src} -> {dst}")
        else:
            print(f"[warn] Missing strip: {src}")


def slice_strips_1x16():
    """
    Each SyncDreamer output (sid.png) is a 1×16 strip:
        [view0][view1]...[view15]

    We cut horizontally into 16 equal tiles per strip, then save the
    first MAX_VIEWS tiles total as view_00.png ... view_XX.png,
    upscaled to UPSCALE_SIZE.
    """
    saved = 0
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for sid in range(NUM_SAMPLES):
        strip_path = OUTPUT_DIR / f"{sid}.png"
        if not strip_path.exists():
            print(f"[warn] Strip not found: {strip_path}, skipping")
            continue

        strip = Image.open(strip_path).convert("RGBA")
        W, H = strip.size

        # 1 row, 16 columns
        cols = VIEWS_PER_STRIP
        tile_w = W // cols
        tile_h = H

        print(f"[info] Slicing strip {strip_path.name} ({W}x{H}) into {cols} tiles")

        for c in range(cols):
            if saved >= MAX_VIEWS:
                print(f"[info] Reached {MAX_VIEWS} views, stopping.")
                return

            left = c * tile_w
            upper = 0
            right = left + tile_w
            lower = tile_h

            tile = strip.crop((left, upper, right, lower))

            # Upscale to desired resolution
            if UPSCALE_SIZE is not None:
                tile = tile.resize(UPSCALE_SIZE, Image.BICUBIC)

            out_path = RESULT_DIR / f"view_{saved:02d}.png"
            tile.save(out_path)
            print(f"[info] Saved {out_path}")
            saved += 1

    print(f"[info] Finished slicing; total views saved: {saved}")


if __name__ == "__main__":
    ensure_rgba()
    run_syncdreamer()
    copy_full_strips()   # keep the raw 1x16 strips
    slice_strips_1x16()  # produce upscaled individual views
    print(f"[info] DONE. Check: {RESULT_DIR}")
