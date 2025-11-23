#!/usr/bin/env python3
"""
generate_3d.py

Use Stability AI's Stable Fast 3D (SF3D) API (v2beta) to turn:
    sketches/0/plain.png  ->  sketches/0/mesh/plain_sf3d.glb

Directory layout (relative to this script):
    InstantMesh/          (ignored)
    sketches/
      0/
        plain.png
    generate_3d.py

No writes go into InstantMesh/, only sketches/0/mesh/.
"""

import os
from pathlib import Path
import requests


# ---------- paths ----------
ROOT = Path(__file__).resolve().parent
INPUT_IMG = ROOT / "sketches" / "0" / "plain.png"
OUTPUT_DIR = ROOT / "sketches" / "0" / "mesh"
OUTPUT_MESH = OUTPUT_DIR / "plain_sf3d.glb"

# ---------- API config ----------
# Get your Stability API key from: https://platform.stability.ai/account/keys
# Either:
#   export STABILITY_API_KEY="sk-xxxxxxxx"
# or hardcode below (not recommended for sharing code).
API_KEY = os.environ.get("STABILITY_API_KEY", "")

# Correct Stable Fast 3D endpoint (REST v2beta)
# Docs / examples all point to this:
#   POST https://api.stability.ai/v2beta/3d/stable-fast-3d
API_URL = "https://api.stability.ai/v2beta/3d/stable-fast-3d"


def main():
    # --- sanity checks ---
    if not API_KEY:
        raise RuntimeError(
            "Missing STABILITY_API_KEY.\n"
            "Set it via `export STABILITY_API_KEY=...` or edit generate_3d.py."
        )

    if not INPUT_IMG.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT_IMG}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parameters for Stable Fast 3D
    # As shown in third-party docs & Stability references: image is required,
    # optional fields include texture_resolution, foreground_ratio, remesh. :contentReference[oaicite:1]{index=1}
    form_data = {
        "texture_resolution": "1024",  # "512", "1024", or "2048"
        "foreground_ratio": "0.85",    # 0.0–1.0, how tight to crop
        # "remesh": "none",           # you can add this if needed
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        # The API returns a GLB binary; we can optionally hint:
        # "Accept": "application/octet-stream",
    }

    suffix = INPUT_IMG.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    else:
        mime = "image/jpeg"

    print(f"[SF3D] Sending {INPUT_IMG} to Stable Fast 3D (v2beta)...")

    with open(INPUT_IMG, "rb") as f:
        files = {
            "image": (INPUT_IMG.name, f, mime),
        }
        resp = requests.post(API_URL, headers=headers, files=files, data=form_data)

    # --- handle response ---
    if resp.status_code != 200:
        print(f"[SF3D] Error {resp.status_code}")
        try:
            print("[SF3D] Response JSON:", resp.json())
        except Exception:
            print("[SF3D] Response text:", resp.text[:500])
        resp.raise_for_status()

    # Response is GLB binary (glTF) in body (arraybuffer in JS examples). :contentReference[oaicite:2]{index=2}
    with open(OUTPUT_MESH, "wb") as out_f:
        out_f.write(resp.content)

    print(f"[SF3D] Mesh saved to: {OUTPUT_MESH}")


if __name__ == "__main__":
    main()
