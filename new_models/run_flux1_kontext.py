#!/usr/bin/env python
"""
run_flux1_kontext.py

Use FLUX.1-Kontext-dev (local, via diffusers) to generate 8 multi-view
versions of a single sketch using only text prompts.

Design goals:
    - Input is a hand-drawn sketch (0.png).
    - Keep the EXACT drawing style: same lines, sketchiness, abstraction level.
    - NO extra realism, NO shading, NO color, NO cleanup.
    - Only change the camera viewpoint in 3D.
    - Default: num_inference_steps = 18 to avoid over-cooking.

Usage:
    python run_flux1_kontext.py
    python run_flux1_kontext.py --input 0.png --outdir flux1_views --steps 18 --guidance 2.5
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import FluxKontextPipeline


# ---------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------

def load_pipeline(device: str = "cuda") -> FluxKontextPipeline:
    """
    Load FLUX.1-Kontext-dev with reduced VRAM usage.

    - Uses float16 instead of bfloat16.
    - Uses accelerate's CPU offload so we don't keep the whole model on GPU.
    """
    model_id = "black-forest-labs/FLUX.1-Kontext-dev"

    print(f"[flux1-kontext] Loading pipeline from {model_id} ...")
    pipe = FluxKontextPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # slightly smaller than bfloat16
    )

    # Important: do NOT call `pipe.to("cuda")` when using offload.
    if device == "cuda":
        # Move modules to GPU just-in-time during forward passes.
        pipe.enable_model_cpu_offload()
        # Optional extra: slice attention to reduce peak memory.
        try:
            pipe.enable_attention_slicing("max")
            print("[flux1-kontext] Enabled attention slicing ('max').")
        except Exception as e:
            print(f"[flux1-kontext] Could not enable attention slicing: {e}")
        print("[flux1-kontext] Enabled model CPU offload.")
    else:
        pipe.to(device)

    return pipe


# ---------------------------------------------------------------------
# View prompts
# ---------------------------------------------------------------------

def build_view_prompts():
    """
    Return a dict: view_name -> prompt string.

    All prompts scream:
        - keep original sketch style
        - no color / shading
        - change ONLY camera/viewpoint
    """

    base = (
        "You are editing a single existing black-and-white line sketch of an object. "
        "In the new image, you MUST preserve the EXACT same drawing style as the original: "
        "same line thickness, same wobble and sketchiness, same level of abstraction, "
        "same black lines on white background, no extra cleanup. "
        "Do NOT add color. Do NOT add shading. Do NOT make it more realistic. "
        "Do NOT invent new decorative details. "
        "Only change the 3D camera viewpoint of the object. "
    )

    views = {
        "front": base
        + "Reproject the SAME object into a PURE FRONT view: the object faces the viewer directly. "
          "Center the object in the frame. Keep all proportions and part relationships consistent with the original sketch.",

        "back": base
        + "Reproject the SAME object into a PURE BACK view: rotate the object 180 degrees around the vertical axis "
          "compared to a front view, so the viewer sees the back side. "
          "Do not change line style or complexity.",

        "left": base
        + "Reproject the SAME object into a PURE LEFT-SIDE orthographic view: the camera is exactly on the object's left side. "
          "Left and right must be interpreted in a consistent world frame, not randomly mirrored. "
          "Show the left profile only, with no perspective distortion.",

        "right": base
        + "Reproject the SAME object into a PURE RIGHT-SIDE orthographic view: the camera is exactly on the object's right side. "
          "This is the mirror counterpart of the left-side view around the vertical axis. "
          "Again, preserve the exact sketch style and line quality.",

        "top": base
        + "Reproject the SAME object into a TOP-DOWN orthographic view: the camera looks straight down from above. "
          "Show only what would be visible from above; keep the object centered.",

        "bottom": base
        + "Reproject the SAME object into a BOTTOM-UP orthographic view: the camera looks straight up from below. "
          "Show only what would be visible from underneath. Keep sketch style exactly unchanged.",

        "front_left_iso": base
        + "Reproject the SAME object into a 3/4 FRONT-LEFT isometric-like view: "
          "the camera is slightly above, between the front and left sides, showing both faces at once. "
          "Use only a gentle perspective; do not exaggerate. Preserve the exact line style.",

        "front_right_iso": base
        + "Reproject the SAME object into a 3/4 FRONT-RIGHT isometric-like view: "
          "the camera is slightly above, between the front and right sides, showing both faces at once. "
          "Again, do not change the drawing style or make the sketch more realistic."
    }

    return views


# ---------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------

def load_input_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    img = Image.open(path).convert("RGB")
    return img


def generate_views(
    pipe: FluxKontextPipeline,
    input_image: Image.Image,
    prompts: dict,
    outdir: Path,
    num_steps: int,
    guidance_scale: float,
    seed: int | None,
    height: int | None,
    width: int | None,
):
    outdir.mkdir(parents=True, exist_ok=True)

    generator = None
    if seed is not None:
        # For offload, keep the generator on CUDA if available; offload handles module placement
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

    for view_name, prompt in prompts.items():
        print(f"[flux1-kontext] Generating view: {view_name} ...")

        kwargs = dict(
            image=input_image,
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        # Only override resolution if explicitly requested (>0)
        if height is not None and height > 0:
            kwargs["height"] = height
        if width is not None and width > 0:
            kwargs["width"] = width

        result = pipe(**kwargs)
        img = result.images[0]
        out_path = outdir / f"{view_name}.png"
        img.save(out_path)
        print(f"[flux1-kontext] Saved: {out_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-view sketch editing with FLUX.1-Kontext-dev"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="0.png",
        help="Input sketch file (default: 0.png)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="flux1_views",
        help="Output directory for generated views",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of inference steps (default: 18 to avoid over-realism)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=2.0,
        help="Guidance scale (lower leans more on input image; default 2.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (set <0 for random each time)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Optional output height. 0 = keep original image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Optional output width. 0 = keep original image width.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[warning] CUDA not available, running on CPU will be extremely slow.")

    seed = None if args.seed is None or args.seed < 0 else args.seed
    height = None if args.height is None or args.height <= 0 else args.height
    width = None if args.width is None or args.width <= 0 else args.width

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    print(f"[flux1-kontext] Loading input sketch: {input_path}")
    input_image = load_input_image(input_path)

    print("[flux1-kontext] Loading FLUX.1-Kontext pipeline...")
    pipe = load_pipeline(device=device)

    prompts = build_view_prompts()

    print(
        f"[flux1-kontext] Generating {len(prompts)} views "
        f"(steps={args.steps}, guidance={args.guidance}, "
        f"height={height or 'orig'}, width={width or 'orig'})..."
    )

    generate_views(
        pipe=pipe,
        input_image=input_image,
        prompts=prompts,
        outdir=outdir,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        seed=seed,
        height=height,
        width=width,
    )

    print("[flux1-kontext] Done.")


if __name__ == "__main__":
    main()
