#!/usr/bin/env python
"""
run_flux2.py

Use FLUX.2-dev (diffusers/FLUX.2-dev-bnb-4bit) with a remote text encoder
to generate 8 multi-view images from a single sketch/image.

Default behavior:
    - Look for "0.png" in the current directory.
    - Create 8 views in ./flux2_views/
    - Style = "realistic"

Usage:
    python run_flux2.py
    python run_flux2.py --input 0.png --style sketch --outdir views_flux2 --height 768 --width 768
"""

import argparse
import io
from pathlib import Path

import requests
import torch
from PIL import Image
from diffusers import Flux2Pipeline
from huggingface_hub import get_token


# ---------- remote text encoder ----------

REMOTE_TEXT_ENCODER_URL = "https://remote-text-encoder-flux-2.huggingface.co/predict"


def remote_text_encoder(prompt: str, device: str = "cuda"):
    """
    Call the remote text encoder recommended in the FLUX.2-dev model card.

    Returns a torch.Tensor on the requested device.
    """
    headers = {
        "Authorization": f"Bearer {get_token()}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        REMOTE_TEXT_ENCODER_URL,
        json={"prompt": prompt},
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    # HF service returns a torch-saved tensor
    prompt_embeds = torch.load(io.BytesIO(resp.content))
    return prompt_embeds.to(device)


# ---------- pipeline loading ----------

def load_pipeline(device: str = "cuda") -> Flux2Pipeline:
    """
    Load FLUX.2-dev-bnb-4bit using the official diffusers pattern:
    - 4-bit quantized DiT + VAE locally
    - text_encoder=None (we use remote_text_encoder instead)
    """
    repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
    torch_dtype = torch.bfloat16

    print(f"[flux2] from_pretrained({repo_id}, dtype={torch_dtype}, text_encoder=None)")
    pipe = Flux2Pipeline.from_pretrained(
        repo_id,
        text_encoder=None,
        torch_dtype=torch_dtype,
    ).to(device)

    return pipe


# ---------- prompts ----------

def build_view_prompts(style: str):
    """
    Return a dict: view_name -> prompt string.

    style: "realistic" or "sketch"
    """

    if style == "sketch":
        base = (
            "Draw the SAME object as in the input reference image, "
            "as a clean black line-art technical sketch on white background, "
            "no shading, no text, no background clutter. "
        )
    else:
        # photoreal
        base = (
            "Render the SAME object as in the input reference image, "
            "as a clean product photo on a plain white background, "
            "high-quality, sharp details, no text, studio lighting. "
        )

    views = {
        "front": base + "Camera directly in FRONT of the object.",
        "left": base + "Camera on the LEFT side, pure side view, orthographic-like.",
        "right": base + "Camera on the RIGHT side, pure side view, orthographic-like.",
        "front_left_iso": base
        + "Camera in a 3/4 FRONT-LEFT isometric view, slightly above.",
        "front_right_iso": base
        + "Camera in a 3/4 FRONT-RIGHT isometric view, slightly above.",
    }
    return views


# ---------- image helpers ----------

def load_input_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    img = Image.open(path).convert("RGB")
    return img


def generate_views(
    pipe: Flux2Pipeline,
    input_image: Image.Image,
    prompts: dict,
    outdir: Path,
    height: int,
    width: int,
    num_steps: int,
    guidance_scale: float,
    seed: int | None = 42,
    device: str = "cuda",
):
    outdir.mkdir(parents=True, exist_ok=True)

    generator = (
        torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    )

    # FLUX.2 expects a *list* of reference images for multi-ref; we pass [input_image]
    ref_images = [input_image]

    for view_name, prompt in prompts.items():
        print(f"[flux2] Generating view: {view_name} ...")
        prompt_embeds = remote_text_encoder(prompt, device=device)

        result = pipe(
            prompt_embeds=prompt_embeds,
            image=ref_images,  # image-to-image conditioning via list
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        img = result.images[0]
        out_path = outdir / f"{view_name}.png"
        img.save(out_path)
        print(f"[flux2] Saved: {out_path}")


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-view generation with FLUX.2-dev")

    parser.add_argument(
        "--input",
        type=str,
        default="0.png",
        help="Input image file (default: 0.png)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="flux2_views",
        help="Output directory for generated views",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="realistic",
        choices=["realistic", "sketch"],
        help="Output style: realistic or sketch-like",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Output image height (pixels)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Output image width (pixels)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=48,
        help="Number of inference steps (28 is a good trade-off)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=4.0,
        help="Guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (set to -1 for random each time)",
    )

    return parser.parse_args()


# ---------- main ----------

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[warning] CUDA not available, running on CPU will be extremely slow.")

    seed = None if args.seed is None or args.seed < 0 else args.seed

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    print(f"[flux2] Loading input image: {input_path}")
    input_image = load_input_image(input_path)

    print("[flux2] Loading FLUX.2 pipeline (this may take a bit on first run)...")
    pipe = load_pipeline(device=device)

    prompts = build_view_prompts(args.style)

    print(
        f"[flux2] Generating {len(prompts)} views "
        f"(style={args.style}, size={args.width}x{args.height})..."
    )
    generate_views(
        pipe=pipe,
        input_image=input_image,
        prompts=prompts,
        outdir=outdir,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        seed=seed,
        device=device,
    )
    print("[flux2] Done.")


if __name__ == "__main__":
    main()
