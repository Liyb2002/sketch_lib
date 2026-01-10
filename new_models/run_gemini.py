#!/usr/bin/env python3
"""
run_gemini_nanabanana.py

Multi-view generation using Gemini image models (“Nano Banana / Nano Banana Pro”).

Behavior:
- Input: one reference image (default: 0.png)
- Output: per-view PNGs in an output folder (default: ./gemini_views/)
- Views (added): front, back, left, right, top, bottom,
                 front_left_iso, front_right_iso, back_left_iso, back_right_iso
- Default style: sketch  (per your request)

Auth:
- If using Vertex AI:
    export GOOGLE_GENAI_USE_VERTEXAI=True
    export GOOGLE_CLOUD_PROJECT=your-project-id
    export GOOGLE_CLOUD_LOCATION=global   (or your region)
- Else (Developer API):
    export GEMINI_API_KEY=...

Usage:
  python run_gemini_nanabanana.py
  python run_gemini_nanabanana.py --input 0.png --style realistic --outdir gemini_views_realistic
  python run_gemini_nanabanana.py --model gemini-2.5-flash-image
"""

import argparse
import os
from io import BytesIO
from pathlib import Path
from typing import Dict

from PIL import Image

from google import genai
from google.genai.types import GenerateContentConfig, Modality


# -------------------------
# Prompts (same naming as FLUX2 + expanded views)
# -------------------------

def build_view_prompts(style: str) -> Dict[str, str]:
    """
    Return dict: view_name -> prompt string.
    style: "realistic" or "sketch"
    """

    if style == "sketch":
        base = (
            "Draw the SAME object as in the input reference image, "
            "as a clean black line-art technical sketch on white background, "
            "no shading, no text, no background clutter. "
            "Preserve exact contour and proportions and drawing style. "
            "Just imagine you are rotating the exact sketch drawing on 2D space. "
        )
    else:
        base = (
            "Render the SAME object as in the input reference image, "
            "as a clean product photo on a plain white background, "
            "high-quality, sharp details, no text, studio lighting. "
            "Preserve exact contour and proportions. "
        )

    # Expanded set (front/side/back/top/bottom + isometric)
    return {
        "front": base + "Camera directly in FRONT of the object.",
        "back": base + "Camera directly BEHIND the object (rear view).",
        "left": base + "Camera on the LEFT side, pure side view, orthographic-like.",
        "right": base + "Camera on the RIGHT side, pure side view, orthographic-like.",
        "top": base + "Camera directly ABOVE the object (top-down view), orthographic-like.",
        "bottom": base + "Camera directly BELOW the object (bottom-up view), orthographic-like.",
        "front_left_iso": base + "Camera in a 3/4 FRONT-LEFT isometric view, slightly above.",
        "front_right_iso": base + "Camera in a 3/4 FRONT-RIGHT isometric view, slightly above.",
        "back_left_iso": base + "Camera in a 3/4 BACK-LEFT isometric view, slightly above.",
        "back_right_iso": base + "Camera in a 3/4 BACK-RIGHT isometric view, slightly above.",
    }


# -------------------------
# IO helpers
# -------------------------

def load_input_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    return Image.open(path).convert("RGB")


def save_first_image_from_response(resp, out_path: Path) -> None:
    """
    Gemini image-capable models return interleaved text+image parts.
    Save the first returned image (inline_data) to out_path.
    """
    if not resp.candidates:
        raise RuntimeError("No candidates returned by the model.")

    parts = resp.candidates[0].content.parts if resp.candidates[0].content else []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is not None and getattr(inline, "data", None):
            img = Image.open(BytesIO(inline.data))
            img.save(out_path)
            return

    # If we got here, no image part was returned
    text_chunks = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
    raise RuntimeError(
        "No image returned by model. Text output was:\n" + "\n".join(text_chunks[:20])
    )


# -------------------------
# Client
# -------------------------

def make_client() -> genai.Client:
    """
    Create a genai.Client() that works in both modes:
    - Vertex AI if GOOGLE_GENAI_USE_VERTEXAI=True (or if project vars exist)
    - Otherwise Gemini Developer API with GEMINI_API_KEY / GOOGLE_API_KEY
    """
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
    has_vertex_vars = bool(os.getenv("GOOGLE_CLOUD_PROJECT")) and bool(os.getenv("GOOGLE_CLOUD_LOCATION"))

    if use_vertex or has_vertex_vars:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        if not project:
            raise RuntimeError(
                "Vertex mode requested but GOOGLE_CLOUD_PROJECT is not set.\n"
                "Set:\n"
                "  export GOOGLE_GENAI_USE_VERTEXAI=True\n"
                "  export GOOGLE_CLOUD_PROJECT=...\n"
                "  export GOOGLE_CLOUD_LOCATION=global"
            )
        return genai.Client(vertexai=True, project=project, location=location)

    # Developer API mode (SDK auto-reads GEMINI_API_KEY / GOOGLE_API_KEY)
    return genai.Client()


# -------------------------
# Generation
# -------------------------

def generate_views(
    client: genai.Client,
    model: str,
    input_image: Image.Image,
    prompts: Dict[str, str],
    outdir: Path,
    temperature: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = GenerateContentConfig(
        response_modalities=[Modality.TEXT, Modality.IMAGE],
        temperature=temperature,
    )

    for view_name, prompt in prompts.items():
        print(f"[gemini] Generating view: {view_name} (model={model}) ...")

        # Image-conditioned generation: provide reference image + instruction.
        resp = client.models.generate_content(
            model=model,
            contents=[input_image, prompt],
            config=cfg,
        )

        out_path = outdir / f"{view_name}.png"
        save_first_image_from_response(resp, out_path)
        print(f"[gemini] Saved: {out_path}")


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Multi-view generation using Gemini image models (Nano Banana / Pro).")

    p.add_argument("--input", type=str, default="0.png", help="Input image file (default: 0.png)")
    p.add_argument("--outdir", type=str, default="gemini_views", help="Output directory")
    # (2) default to sketch
    p.add_argument("--style", type=str, default="sketch", choices=["realistic", "sketch"], help="Output style")

    # Default to “Nano Banana Pro”-ish model name; override with --model if needed.
    p.add_argument(
        "--model",
        type=str,
        default="gemini-3-pro-image-preview",
        help="Image model (default: gemini-3-pro-image-preview). Alternatives: gemini-2.5-flash-image",
    )

    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (lower = steadier)")

    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    print(f"[gemini] Loading input image: {input_path}")
    input_image = load_input_image(input_path)

    prompts = build_view_prompts(args.style)

    print("[gemini] Creating client ...")
    client = make_client()

    print(f"[gemini] Generating {len(prompts)} views (style={args.style}) into: {outdir}")
    generate_views(
        client=client,
        model=args.model,
        input_image=input_image,
        prompts=prompts,
        outdir=outdir,
        temperature=args.temperature,
    )

    print("[gemini] Done.")


if __name__ == "__main__":
    main()
