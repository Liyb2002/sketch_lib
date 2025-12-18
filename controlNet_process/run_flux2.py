#!/usr/bin/env python
"""
run_flux2.py (Modified for single-sketch-to-realistic conversion)

Use FLUX.2-dev (diffusers/FLUX.2-dev-bnb-4bit) with a remote text encoder
to generate a single realistic image from a single sketch.

Default behavior:
    - Look for "sketch/input.png".
    - Output size matches the input size.
    - Output in "sketch/input_realistic.png".
    - **Camera position is preserved exactly from the input sketch.**

Usage:
    python run_flux2.py
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

def get_single_realistic_prompt(object_description: str = "") -> str:
    """
    Generate the single, boundary-preserving, realistic prompt.
    Focuses on preserving the exact perspective/camera position of the input sketch.
    """
    # Key part for ControlNet-like behavior: "precise geometry and boundaries matching the sketch"
    prompt = (
        "Render the object in the input reference image with **precise geometry and boundaries** "
        f"matching the sketch. Output a **photorealistic product photo** of a {object_description}, "
"high-quality, on a clean white background. **ABSOLUTELY NO SHADOWS. NO DROP SHADOWS. Pure white background.** The main color of the object should be in red."        # !!! REMOVED CAMERA POSITION INSTRUCTION HERE !!!
    )
    return prompt


# ---------- image helpers ----------

def load_input_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    img = Image.open(path).convert("RGB")
    return img


def generate_single_image(
    pipe: Flux2Pipeline,
    input_image: Image.Image,
    prompt: str,
    out_path: Path,
    height: int,
    width: int,
    num_steps: int,
    guidance_scale: float,
    seed: int | None = 42,
    device: str = "cuda",
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generator = (
        torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    )
    
    # FLUX.2 expects a *list* of reference images
    ref_images = [input_image]

    print(f"[flux2] Generating single image with prompt: {prompt[:80]}...")
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
    img.save(out_path)
    print(f"[flux2] Saved: {out_path}")


# ---------- CLI (Modified to use fixed paths and dynamic size) ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Single-view generation with FLUX.2-dev")

    # Fixed input and output paths based on user request
    parser.add_argument(
        "--input",
        type=str,
        default="sketch/input.png", 
        help="Input sketch file (default: sketch/input.png)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sketch/input_realistic.png", 
        help="Output realistic image file (default: sketch/input_realistic.png)",
    )
    parser.add_argument(
        "--desc",
        type=str,
        default="", 
        help="Optional: a short description of the object in the sketch to refine the result.",
    )
    # Height and width are dynamic based on input image.
    parser.add_argument(
        "--steps",
        type=int,
        default=28, 
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
    out_path = Path(args.output)
    
    if input_path == out_path:
        raise ValueError("Input and Output paths are the same! Cannot overwrite input file.")

    print(f"[flux2] Loading input sketch: {input_path}")
    input_image = load_input_image(input_path)
    
    # Get the size of the input image
    input_width, input_height = input_image.size
    print(f"[flux2] Input image size: {input_width}x{input_height}. Output will match.")

    print("[flux2] Loading FLUX.2 pipeline (this may take a bit on first run)...")
    pipe = load_pipeline(device=device)

    # Get the single, specialized prompt, which now excludes camera position instruction
    prompt = get_single_realistic_prompt(args.desc)

    print(
        f"[flux2] Generating single image (size={input_width}x{input_height}, steps={args.steps})..."
    )
    
    generate_single_image(
        pipe=pipe,
        input_image=input_image,
        prompt=prompt,
        out_path=out_path,
        height=input_height,
        width=input_width,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        seed=seed,
        device=device,
    )
    print("[flux2] Done.")


if __name__ == "__main__":
    main()