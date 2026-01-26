#!/usr/bin/env python3
# 15_image_fix_corrupted_gemini.py
#
# For each view_x:
#   input:  sketch/final_outputs/view_x/fix/corrupted.png
#   output: sketch/final_outputs/view_x/gemini_fix/raw.png
#           sketch/final_outputs/view_x/gemini_fix/final.png
#
# IMPORTANT ENFORCEMENT:
# - Define editable region as RED pixels in the INPUT corrupted.png (not Gemini output).
# - final.png = composite:
#     * outside red region: take INPUT exactly
#     * inside red region:  take GEMINI, but if GEMINI pixel is red-ish => set to WHITE
#
# Env:
#   export GEMINI_API_KEY="..."

import os
import glob
import base64
import json
import time
from io import BytesIO
from typing import Optional

import requests
import cv2
import numpy as np
from PIL import Image


MODEL = "gemini-2.5-flash-image"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

PROMPT = (
    "You are given one image.\n\n"
    "This image is a hand-drawn style sketch created by blending two separate design pieces together.\n\n"
    "Some regions in the image are deliberately marked in pure red. These red regions indicate corrupted blending "
    "boundaries between the two pieces.\n\n"
    "Your task is to repair the sketch so that the two pieces form a single clean, continuous drawing.\n\n"
    "Rules and goals:\n"
    "1) Meaning of red regions\n"
    "- Red pixels mark the ONLY areas that need fixing.\n"
    "- They indicate seams, gaps, overlaps, broken strokes, or misaligned boundaries caused by blending.\n\n"
    "2) What to do inside red regions\n"
    "- Reconnect broken lines smoothly.\n"
    "- Close small gaps and holes.\n"
    "- Remove doubled or overlapping strokes.\n"
    "- Align contours so edges flow naturally across the boundary.\n"
    "- Keep the sketch style consistent with the surrounding drawing (thin black hand-drawn lines on white background).\n\n"
    "3) What NOT to do\n"
    "- Do NOT move, resize, or redesign the components.\n"
    "- Do NOT introduce new parts or decorative details.\n"
    "- Do NOT change the overall structure or proportions.\n\n"
    "4) Output requirements\n"
    "- Output a single repaired image.\n"
    "- The final image should look like a clean, continuous sketch with no visible seams or red marks.\n"
    "- The red color must be completely removed and replaced by natural sketch strokes that blend with neighboring lines.\n\n"
    "Think of this as surgically repairing the seam between two overlaid sketches, not redrawing the design."
)


def b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def make_full_white_mask_png_bytes(w: int, h: int) -> bytes:
    img = Image.fromarray(np.full((h, w), 255, dtype=np.uint8), mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def call_gemini_image_edit(api_key: str, image_b64: str, mask_b64: str, prompt: str) -> bytes:
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                {"inline_data": {"mime_type": "image/png", "data": mask_b64}},
            ]
        }],
        "generationConfig": {"responseModalities": ["IMAGE"]}
    }

    r = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:2000]}")

    j = r.json()
    cands = j.get("candidates", [])
    if not cands:
        raise RuntimeError("No candidates in Gemini response")

    parts = (((cands[0] or {}).get("content") or {}).get("parts")) or []
    for p in parts:
        inline = p.get("inline_data") or p.get("inlineData")
        if not inline:
            continue
        data = inline.get("data")
        mime = inline.get("mime_type") or inline.get("mimeType") or ""
        if isinstance(data, str) and len(data) > 100 and ("image" in mime or mime == ""):
            return base64.b64decode(data)

    raise RuntimeError("No image payload found in Gemini response")


def decode_png_bytes_to_bgr(png_bytes: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr
    rgb = np.array(Image.open(BytesIO(png_bytes)).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def red_mask_from_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Return boolean mask where pixels are 'red' (in HSV).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 80], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 120, 80], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    return (m1 | m2) > 0


def enforce_red_region_only(input_bgr: np.ndarray, gemini_bgr: np.ndarray) -> np.ndarray:
    """
    final rule:
      - outside RED region (computed from input): output = input
      - inside  RED region: output = gemini, but any red-ish pixel in gemini => white
    """
    H, W = input_bgr.shape[:2]

    if gemini_bgr.shape[:2] != (H, W):
        gemini_bgr = cv2.resize(gemini_bgr, (W, H), interpolation=cv2.INTER_AREA)

    edit = red_mask_from_bgr(input_bgr)  # editable area from INPUT red

    out = input_bgr.copy()

    # inside editable region: start with Gemini
    out[edit] = gemini_bgr[edit]

    # but if Gemini still outputs red inside editable region => white it out
    gem_red = red_mask_from_bgr(gemini_bgr)
    red_inside = edit & gem_red
    out[red_inside] = (255, 255, 255)

    return out


def main():
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")

    view_dirs = sorted(glob.glob(os.path.join("sketch", "final_outputs", "view_*")))
    if not view_dirs:
        raise FileNotFoundError("No sketch/final_outputs/view_* folders found")

    for view_dir in view_dirs:
        view_name = os.path.basename(view_dir)
        corrupted_png = os.path.join(view_dir, "fix", "corrupted.png")

        if not os.path.exists(corrupted_png):
            print(f"[SKIP] {view_name}: no fix/corrupted.png")
            continue

        input_bgr = cv2.imread(corrupted_png, cv2.IMREAD_COLOR)
        if input_bgr is None:
            print(f"[SKIP] {view_name}: failed to read {corrupted_png}")
            continue

        H, W = input_bgr.shape[:2]

        # We still send a full-white mask; enforcement happens locally.
        mask_bytes = make_full_white_mask_png_bytes(W, H)
        mask_b64 = base64.b64encode(mask_bytes).decode("utf-8")
        img_b64 = b64_file(corrupted_png)

        out_dir = os.path.join(view_dir, "gemini_fix")
        raw_path = os.path.join(out_dir, "raw.png")
        final_path = os.path.join(out_dir, "final.png")
        os.makedirs(out_dir, exist_ok=True)

        print(f"[PROC] {view_name}  input={corrupted_png}")

        raw_bytes: Optional[bytes] = None
        last_err: Optional[Exception] = None

        for attempt in range(3):
            try:
                raw_bytes = call_gemini_image_edit(api_key, img_b64, mask_b64, PROMPT)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(1.0 + attempt)

        if last_err is not None:
            print(f"  ! FAILED: {last_err}")
            continue

        assert raw_bytes is not None
        write_bytes(raw_path, raw_bytes)

        gem_bgr = decode_png_bytes_to_bgr(raw_bytes)
        final_bgr = enforce_red_region_only(input_bgr, gem_bgr)
        cv2.imwrite(final_path, final_bgr)

        print(f"  saved: {raw_path}")
        print(f"  saved: {final_path}")

    print("Done.")


if __name__ == "__main__":
    main()
