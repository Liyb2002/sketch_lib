import torch
import os
import glob
import io
import requests
from PIL import Image
from diffusers import Flux2Pipeline
from huggingface_hub import get_token

# --- Configuration ---
INPUT_DIR = "sketch/views"
OUTPUT_DIR = "sketch/ctrl_views"
HF_TOKEN = "your_huggingface_token_here" # Paste your token here

PROMPT = (
    "A professional, high-end studio product photograph of a modern [PRODUCT]. "
    "Features premium matte textures and sharp industrial edges. "
    "Completely isolated on a solid, pure white background (#FFFFFF). "
    "Clean commercial lighting, 8k resolution, photorealistic."
)

# Helper function to use the remote encoder (saves ~15GB VRAM)
def remote_text_encoder(prompts):
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompts},
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
    )
    # The API returns the raw torch tensor embeddings
    return torch.load(io.BytesIO(response.content))

def process_sketches():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda"
    
    # 1. Load the pre-quantized FLUX.2 Transformer
    # This repo is specifically designed to fit on 16GB-24GB cards
    pipe = Flux2Pipeline.from_pretrained(
        "diffusers/FLUX.2-dev-bnb-4bit", 
        text_encoder=None, # We use the remote encoder instead
        torch_dtype=torch.bfloat16
    ).to(device)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "view_*.png")))

    for img_path in files:
        filename = os.path.basename(img_path)
        sketch_image = Image.open(img_path).convert("RGB").resize((1024, 1024))

        print(f"Encoding prompt and rendering: {filename}...")
        
        # Get embeddings from the cloud so your GPU stays empty for the image
        prompt_embeds = remote_text_encoder(PROMPT).to(device)

        with torch.inference_mode():
            output = pipe(
                prompt_embeds=prompt_embeds,
                image=sketch_image, # This is the "Reference" image (your sketch)
                num_inference_steps=20,
                guidance_scale=4.0,
                generator=torch.Generator(device=device).manual_seed(42),
            ).images[0]

        # Post-process for perfect white background
        np_img = np.array(output)
        np_img[np.all(np_img > 245, axis=-1)] = [255, 255, 255]
        
        Image.fromarray(np_img).save(os.path.join(OUTPUT_DIR, filename))

if __name__ == "__main__":
    process_sketches()