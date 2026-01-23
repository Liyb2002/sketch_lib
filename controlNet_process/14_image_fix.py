#!/usr/bin/env python3
# 14_image_fix.py
#
# Gemini API-key masked "inpainting" via Nano Banana (Gemini 2.5 Flash Image).
# We provide (new.png, mask.png) as two images and instruct the model:
#   white mask = editable, black mask = keep identical.
#
# Inputs per view:
#   sketch/final_outputs/view_x/gemini_fix/new.png
#   sketch/final_outputs/view_x/gemini_fix/mask.png
#
# Outputs:
#   sketch/final_outputs/view_x/gemini_fix/gemini_fixed.png
#   sketch/final_outputs/view_x/gemini_fixed.png

import os
import glob
import base64
import json
import time
from typing import Optional

import requests


MODEL = "gemini-2.5-flash-image"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

PROMPT = (
    "You are repairing compositing artifacts caused by translating/scaling components in a CAD-like render.\n"
    "You are given two images:\n"
    "  (1) The input image to fix.\n"
    "  (2) A binary mask image of the same size.\n\n"
    "Mask semantics:\n"
    "- White pixels in the mask (value near 255) are the ONLY region you may change.\n"
    "- Black pixels in the mask (value near 0) must remain IDENTICAL to the input image.\n\n"
    "Task inside white mask ONLY:\n"
    "- Fix boundary seams, small gaps, overlaps, jagged edges, and discontinuities.\n"
    "- Blend edges/shading so neighboring regions look continuous and natural.\n\n"
    "Hard rules:\n"
    "- Do NOT move any components.\n"
    "- Do NOT change layout/geometry.\n"
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

    # Two images as inline_data parts + text instruction.
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                {"inline_data": {"mime_type": "image/png", "data": mask_b64}},
            ]
        }],
        # Ensure image output
        "generationConfig": {
            "responseModalities": ["IMAGE"]
        }
    }

    r = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=300)

    # Helpful error printing (Gemini often returns JSON error)
    if r.status_code != 200:
        txt = (r.text or "").strip()
        raise RuntimeError(f"HTTP {r.status_code}: {txt[:2000]}")

    j = r.json()

    # Parse returned image bytes:
    # candidates[0].content.parts[*].inline_data.data
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


def main():
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Set env var GEMINI_API_KEY (or GOOGLE_API_KEY).")

    view_dirs = sorted(glob.glob(os.path.join("sketch", "final_outputs", "view_*")))
    if not view_dirs:
        raise FileNotFoundError("No folders found: sketch/final_outputs/view_*")

    for view_dir in view_dirs:
        view_name = os.path.basename(view_dir)
        fix_dir = os.path.join(view_dir, "gemini_fix")
        new_png = os.path.join(fix_dir, "new.png")
        mask_png = os.path.join(fix_dir, "mask.png")

        if not (os.path.exists(new_png) and os.path.exists(mask_png)):
            continue

        out_in_fix = os.path.join(fix_dir, "gemini_fixed.png")
        out_in_view = os.path.join(view_dir, "gemini_fixed.png")

        img_b64 = b64_file(new_png)
        mask_b64 = b64_file(mask_png)

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                out_bytes = call_gemini_image_edit(api_key, img_b64, mask_b64, PROMPT)
                write_bytes(out_in_fix, out_bytes)
                write_bytes(out_in_view, out_bytes)
                print(f"[OK] {view_name} -> {out_in_view}")
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
