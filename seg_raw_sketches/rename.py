#!/usr/bin/env python3
"""
Rename all images inside sketches/<object_name>/ to:
    0.png, 1.png, 2.png, ...
"""

from pathlib import Path

ROOT = Path("sketches")  # your root folder

VALID_SUFFIXES = [".png", ".jpg", ".jpeg"]


def rename_in_folder(folder: Path):
    print(f"Renaming images in {folder}")

    imgs = [p for p in folder.iterdir() if p.suffix.lower() in VALID_SUFFIXES]
    imgs = sorted(imgs)  # stable order

    for i, p in enumerate(imgs):
        new_path = folder / f"{i}.png"
        p.rename(new_path)

    print(f"  -> renamed {len(imgs)} images")


def main():
    # go into each subfolder inside sketches
    for obj_dir in ROOT.iterdir():
        if obj_dir.is_dir():
            rename_in_folder(obj_dir)


if __name__ == "__main__":
    main()
