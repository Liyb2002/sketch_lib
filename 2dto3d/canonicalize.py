#!/usr/bin/env python3
"""
canonicalize_nanobanana_quarter_view.py

For each sketch image in sketches/{OBJECT_TYPE}, call Gemini
("nano-banana") to redraw it in a canonical view:

    - front-right three-quarter view
    - object facing to the RIGHT side of the image
    - you see the front and the right side
    - camera slightly above the object
    - black lines, white background, same sketch style, no new parts

We:
  - Process images in numeric order: 0.png, 1.png, 2.png, ...
  - Create a NEW Gemini client for EACH IMAGE (your request).
  - Use temperature=0.0 for maximum stability.

Setup:
    pip install google-genai pillow
    export GEMINI_API_KEY="YOUR_KEY"
"""

import os
from pathlib import Path
from io import BytesIO

from PIL import Image
from google import genai
from google.genai import types


# ---------------- CONFIG ---------------- #

SKETCHES_ROOT = Path("sketches")

# Change this per object type, or wrap in a loop over subfolders
OBJECT_TYPE = "chairs"  # e.g. "chairs", "car", "bike", "plane"

# Image-capable Gemini model (adjust if your key uses a different one)
MODEL_NAME = "gemini-2.5-flash-image"


def make_prompt(object_type: str) -> str:
    # Short and explicit about orientation and facing direction.
    return f"""
A {object_type} sketch.
Keep the exact same object. But rotate it for 
Front-right 3/4 view, camera slightly above.
The object is facing to the RIGHT side of the image:
you see its front and its right side.
Black lines only, white background, no new details.
""".strip()


PROMPT = make_prompt(OBJECT_TYPE)


# ------------- GEMINI CLIENT ------------- #

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def redraw_canonical(
    sketch_img: Image.Image,
) -> Image.Image:
    """
    Create a NEW Gemini client, then call it once to redraw sketch_img
    in the canonical view.

    contents = [PROMPT, sketch_img]
    """
    sketch_img = sketch_img.convert("RGB")

    # New client each time, per your request
    client = get_client()

    cfg = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        # Make it as deterministic as possible
        temperature=0.0,
        top_p=0.0,
        top_k=1,
    )

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[PROMPT, sketch_img],
        config=cfg,
    )

    # Newer SDK: response.parts
    if getattr(resp, "parts", None):
        for part in resp.parts:
            if getattr(part, "inline_data", None):
                if hasattr(part, "as_image"):
                    return part.as_image()
                return Image.open(BytesIO(part.inline_data.data))

    # Fallback: older candidates[0].content.parts
    if getattr(resp, "candidates", None):
        cand = resp.candidates[0]
        for part in cand.content.parts:
            if getattr(part, "inline_data", None):
                if hasattr(part, "as_image"):
                    return part.as_image()
                return Image.open(BytesIO(part.inline_data.data))

    raise RuntimeError("No image returned from Gemini image model.")


# ------------- MAIN PIPELINE ------------- #

def process_object_folder(object_folder: Path) -> None:
    """
    For sketches/{OBJECT_TYPE}:

    - Create canonicalized_images/ subfolder.
    - For each *.png, in numeric order, call redraw_canonical() and save result.
    """
    if not object_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {object_folder}")

    out_dir = object_folder / "canonicalized_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Numeric sort: 0.png, 1.png, 2.png, ..., 10.png, etc.
    pngs = sorted(
        [p for p in object_folder.glob("*.png")],
        key=lambda p: int(p.stem) if p.stem.isdigit() else 999999,
    )

    if not pngs:
        print(f"[INFO] No PNGs found in {object_folder}")
        return

    print(f"[INFO] Found {len(pngs)} PNGs in {object_folder}")

    for img_path in pngs:
        print(f"[INFO] Canonicalizing {img_path.name}...")
        sketch = Image.open(img_path)

        try:
            canon = redraw_canonical(sketch)
        except Exception as e:
            print(f"[ERROR] Failed on {img_path.name}: {e}")
            continue

        out_path = out_dir / img_path.name
        canon.save(out_path)
        print(f"[OK] Saved {out_path}")

    print("[DONE]")


def main():
    folder = SKETCHES_ROOT / OBJECT_TYPE
    print(f"[INFO] Processing object type: {OBJECT_TYPE}")
    print(f"[INFO] Folder: {folder}")
    process_object_folder(folder)


if __name__ == "__main__":
    main()
