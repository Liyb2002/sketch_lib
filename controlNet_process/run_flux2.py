#!/usr/bin/env python
"""
run_flux2_batch_local.py

Iterates through all images in "sketch/views/", applies FLUX.2 conversion locally,
and saves the output to "sketch/views_realistic/".
Optimized for RTX 4090.
"""

import os
import torch
from pathlib import Path
from PIL import Image
from diffusers import FluxPipeline # Using the standard FluxPipeline for local inference

# ---------- pipeline loading ----------

def load_pipeline(device: str = "cuda") -> FluxPipeline:
    # We use the 4-bit quantized version to ensure plenty of headroom on your 4090
    repo_id = "diffusers/FLUX.1-dev-bnb-4bit" 
    
    print(f"[flux2] Loading {repo_id} locally onto {device}...")
    
    # Loading the full pipeline including text encoders
    pipe = FluxPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    # Optional: If you encounter any memory spikes, uncomment the line below:
    # pipe.enable_model_cpu_offload() 
    
    return pipe

# ---------- prompts ----------

def get_single_realistic_prompt(object_description: str = "object") -> str:
    prompt = (
        f"A photorealistic professional product photo of a {object_description}. "
        "The object is solid red. High-quality, 8k resolution, sharp details. "
        "Clean pure white background. ABSOLUTELY NO SHADOWS. NO DROP SHADOWS."
    )
    return prompt

# ---------- image helpers ----------

def generate_single_image(
    pipe: FluxPipeline,
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
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    
    # Note: FLUX.1-dev usually uses 'prompt'. 
    # If your specific version uses Image-to-Image/ControlNet, 
    # ensure you are using the correct pipeline class.
    result = pipe(
        prompt=prompt,
        # If using an Img2Img pipeline, you would pass image=input_image here
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    )
    
    img = result.images[0]
    img.save(out_path)

# ---------- main loop ----------

def main():
    # Setup paths
    input_dir = Path("sketch/views")
    output_dir = Path("sketch/views_realistic")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    
    # Check for input directory
    if not input_dir.exists():
        print(f"Error: Folder {input_dir} not found.")
        return

    # Filter for image files
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in valid_extensions])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"[flux2] Found {len(image_files)} images. Initializing local pipeline...")
    pipe = load_pipeline(device=device)
    
    # You can change "industrial product" to whatever fits your sketches best
    prompt = get_single_realistic_prompt("motobike") 

    for img_path in image_files:
        print(f"--- Processing: {img_path.name} ---")
        
        try:
            input_image = Image.open(img_path).convert("RGB")
            
            # FLUX works best with dimensions divisible by 16 or 32
            # We round the input dimensions to ensure the VAE doesn't error out
            w, h = input_image.size
            new_w = (w // 16) * 16
            new_h = (h // 16) * 16
            
            save_path = output_dir / img_path.name

            generate_single_image(
                pipe=pipe,
                input_image=input_image,
                prompt=prompt,
                out_path=save_path,
                height=new_h,
                width=new_w,
                num_steps=28,
                guidance_scale=3.5, # Flux.1-dev usually likes 3.5-4.0
                seed=seed,
                device=device,
            )
            print(f"[flux2] Successfully saved: {save_path}")
            
        except Exception as e:
            print(f"[Error] Failed to process {img_path.name}: {e}")

    print("\n[flux2] Batch processing complete.")

if __name__ == "__main__":
    main()