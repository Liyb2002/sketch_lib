#!/usr/bin/env python
"""
run_flux2_pose.py

Use FLUX.2-dev (diffusers/FLUX.2-dev-bnb-4bit) with a remote text encoder
and TWO reference images:

    - src image: the object whose identity/style we want to preserve (e.g., 0.png)
    - pose image: the image whose pose/camera we want to mimic (e.g., 1.png)

Default behavior:
    python run_flux2_pose.py
        --src 0.png
        --pose 1.png
        --out pose_view.png
        --style realistic

You can also do sketch style:
    python run_flux2_pose.py --style sketch
"""

import argparse
import io
from pathlib import Path

import requests
import torch
from PIL import Image
from diffusers import Flux2Pipeline
from huggingface_hub import get_token


# ---------------------------------------------------------------------
# Remote text encoder (from FLUX.2-dev model card pattern)
# ---------------------------------------------------------------------

REMOTE_TEXT_ENCODER_URL = "https://remote-text-encoder-flux-2.huggingface.co/predict"


def remote_text_encoder(prompt: str, device: str = "cuda") -> torch.Tensor:
    """
    Call the remote text encoder recommended for FLUX.2-dev.

    Returns a prompt_embeds tensor on the requested device.
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
    prompt_embeds = torch.load(io.BytesIO(resp.content))
    return prompt_embeds.to(device)


# ---------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------

def load_pipeline(device: str = "cuda") -> Flux2Pipeline:
    """
    Load FLUX.2-dev-bnb-4bit using diffusers.
    We set text_encoder=None (we use the remote encoder).
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


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

def build_pose_prompt(style: str) -> str:
    """
    Build a prompt that tells the model clearly:
    - Identity/style from first reference
    - Pose/camera from second reference
    """

    if style == "sketch":
        base = (
            "Using the FIRST reference image as the object identity and appearance, "
            "and the SECOND reference image as the camera angle and pose, "
            "draw a clean black line-art technical sketch of the FIRST object "
            "in the SAME camera viewpoint and pose as the SECOND image. "
            "White background, no shading, no text, no extra clutter. "
        )
    else:
        base = (
            "Using the FIRST reference image as the object identity and appearance, "
            "and the SECOND reference image as the camera angle and pose, "
            "render a clean high-quality product photo of the FIRST object "
            "in the SAME viewpoint and orientation as the SECOND image. "
            "Plain white studio background, sharp details, no text. "
        )

    return base


# ---------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------

def load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------

def generate_pose_view(
    pipe: Flux2Pipeline,
    src_image: Image.Image,
    pose_image: Image.Image,
    prompt: str,
    out_path: Path,
    height: int,
    width: int,
    num_steps: int,
    guidance_scale: float,
    seed: int | None = 42,
    device: str = "cuda",
):
    # Two references: [identity, pose]
    ref_images = [src_image, pose_image]

    generator = (
        torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    )

    print("[flux2] Getting prompt embeddings from remote text encoder...")
    prompt_embeds = remote_text_encoder(prompt, device=device)

    print("[flux2] Running FLUX.2 sampling...")
    result = pipe(
        prompt_embeds=prompt_embeds,
        image=ref_images,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    )

    img = result.images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[flux2] Saved pose-conditioned view to: {out_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pose-conditioned view generation with FLUX.2-dev"
    )

    parser.add_argument(
        "--src",
        type=str,
        default="1.png",
        help="Source image: object whose identity/style we keep (default: 0.png)",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default="0.png",
        help="Pose image: image whose camera/pose we want to mimic (default: 1.png)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="pose_view.png",
        help="Output image filename (default: pose_view.png)",
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
        help="Output height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Output width",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=18,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=4.5,
        help="Guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed (set <0 for random)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[warning] CUDA not available; CPU will be extremely slow.")

    seed = None if args.seed is None or args.seed < 0 else args.seed

    src_path = Path(args.src)
    pose_path = Path(args.pose)
    out_path = Path(args.out)

    print(f"[flux2] Loading src image (identity): {src_path}")
    src_image = load_image(src_path)

    print(f"[flux2] Loading pose image (camera):  {pose_path}")
    pose_image = load_image(pose_path)

    print("[flux2] Loading FLUX.2 pipeline...")
    pipe = load_pipeline(device=device)

    prompt = build_pose_prompt(args.style)
    print("[flux2] Prompt:")
    print(" ", prompt)

    generate_pose_view(
        pipe=pipe,
        src_image=src_image,
        pose_image=pose_image,
        prompt=prompt,
        out_path=out_path,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        seed=seed,
        device=device,
    )


if __name__ == "__main__":
    main()
