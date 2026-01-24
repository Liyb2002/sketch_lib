#!/usr/bin/env python3
# 14_image_fix.py
#
# Gemini API-key masked "inpainting" via Gemini Image model.
# We provide (input.png, mask.png) as two images and instruct the model:
#   white mask = editable, black mask = keep identical.
#
# Per view:
#   mask:  sketch/final_outputs/view_x/fix/diff_sum_mask.png
#   input: tries (in order):
#          sketch/final_outputs/view_x/gemini_fixed.png
#          sketch/final_outputs/view_x/new.png
#          sketch/final_outputs/view_x/fix/new.png
#          sketch/final_outputs/view_x/fix/input.png
#          sketch/final_outputs/view_x/view.png
#          sketch/view/view_x.png   (fallback)
#
# Output:
#   sketch/final_outputs/view_x/fix/gemini_fixed.png

import os
import glob
import base64
import json
import time
from io import BytesIO
from typing import Optional, List

import requests


MODEL = "gemini-2.5-flash-image"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

PROMPT = (
    "You are repairing compositing artifacts after CAD-like components were translated/scaled.\n"
    "You are given two images:\n"
    "  (1) The input image to fix.\n"
    "  (2) A binary mask image of the same size.\n\n"
    "Mask semantics:\n"
    "- White pixels in the mask (value near 255) are the ONLY region you may change.\n"
    "- Black pixels in the mask (value near 0) must remain EXACTLY the same as the input.\n\n"
    "What the white mask contains:\n"
    "- Only the contours and connection regions between components (seams, borders, small gaps).\n\n"
    "Task (inside white mask ONLY):\n"
    "- Make component connections smoother and more continuous.\n"
    "- Fix boundary seams, tiny gaps/holes, overlaps, jagged edges, and discontinuities.\n"
    "- Blend local edge shading so adjacent regions connect naturally.\n"
    "- Keep the overall shapes unchanged; only refine the contour/connection quality.\n\n"
    "Hard rules:\n"
    "- Do NOT move any components.\n"
    "- Do NOT change geometry/layout.\n"
    "- Do NOT add/remove parts.\n"
    "- Do NOT change colors or lighting globally.\n"
    "- Outside the mask must match the original input pixel-for-pixel.\n\n"
    "Return a single corrected image."
)


def b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def call_gemini_image_edit(api_key: str, image_b64: str, mask_b64: str, prompt: str) -> bytes:
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                {"inline_data": {"mime_type": "image/png", "data": mask_b64}},
            ]
        }],
        "generationConfig": {
            "responseModalities": ["IMAGE"]
        }
    }

    r = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=300)

    if r.status_code != 200:
        txt = (r.text or "").strip()
        raise RuntimeError(f"HTTP {r.status_code}: {txt[:2000]}")

    j = r.json()

    cands = j.get("candidates", [])
    if not cands:
        raise RuntimeError(f"No candidates in response: {json.dumps(j)[:2000]}")

    parts = (((cands[0] or {}).get("content") or {}).get("parts")) or []
    for p in parts:
        inline = p.get("inline_data") or p.get("inlineData")
        if not inline:
            continue
        data = inline.get("data")
        mime = inline.get("mime_type") or inline.get("mimeType") or ""
        if isinstance(data, str) and len(data) > 100 and ("image" in mime or mime == ""):
            return base64.b64decode(data)

    raise RuntimeError(f"Could not find image inline_data in response: {json.dumps(j)[:2000]}")


def enforce_mask_composite(orig_png: str, mask_png: str, gen_bytes: bytes, thresh: int = 128) -> bytes:
    # Hard guarantee: outside-mask stays identical to orig_png
    from PIL import Image
    import numpy as np

    orig = Image.open(orig_png).convert("RGBA")
    mask = Image.open(mask_png).convert("L")
    gen  = Image.open(BytesIO(gen_bytes)).convert("RGBA")

    if gen.size != orig.size:
        gen = gen.resize(orig.size, resample=Image.BILINEAR)
    if mask.size != orig.size:
        mask = mask.resize(orig.size, resample=Image.NEAREST)

    m = np.array(mask, dtype=np.uint8)
    m = (m >= thresh).astype(np.uint8)  # 1 where editable

    o = np.array(orig, dtype=np.uint8)
    g = np.array(gen,  dtype=np.uint8)

    m4 = m[..., None]  # broadcast to RGBA
    out = o * (1 - m4) + g * m4

    out_img = Image.fromarray(out.astype(np.uint8), mode="RGBA")
    buf = BytesIO()
    out_img.save(buf, format="PNG")
    return buf.getvalue()


def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def main():
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Set env var GEMINI_API_KEY (or GOOGLE_API_KEY).")

    view_dirs = sorted(glob.glob(os.path.join("sketch", "final_outputs", "view_*")))
    if not view_dirs:
        raise FileNotFoundError("No folders found: sketch/final_outputs/view_*")

    for view_dir in view_dirs:
        view_name = os.path.basename(view_dir)  # e.g. view_0
        fix_dir = os.path.join(view_dir, "fix")
        mask_png = os.path.join(fix_dir, "diff_sum_mask.png")

        if not os.path.exists(mask_png):
            continue

        # choose an input image to fix
        candidates = [
            os.path.join(view_dir, "gemini_fixed.png"),
            os.path.join(view_dir, "new.png"),
            os.path.join(fix_dir, "new.png"),
            os.path.join(fix_dir, "input.png"),
            os.path.join(view_dir, "view.png"),
            os.path.join("sketch", "view", f"{view_name}.png"),
        ]
        in_png = first_existing(candidates)
        if in_png is None:
            print(f"[SKIP] {view_name}: no input image found (checked: {candidates})")
            continue

        out_path = os.path.join(fix_dir, "gemini_fixed.png")

        img_b64 = b64_file(in_png)
        mask_b64 = b64_file(mask_png)

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                raw_bytes = call_gemini_image_edit(api_key, img_b64, mask_b64, PROMPT)
                out_bytes = enforce_mask_composite(in_png, mask_png, raw_bytes, thresh=128)
                write_bytes(out_path, out_bytes)
                print(f"[OK] {view_name} -> {out_path} (input={os.path.relpath(in_png)})")
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(1.0 + attempt)

        if last_err is not None:
            raise last_err

    print("Done.")


if __name__ == "__main__":
    main()
