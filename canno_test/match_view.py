#!/usr/bin/env python3
"""
match_view.py — Find the canonical view using DINOv2 (Structural Mode).
Features:
- Intermediate Layers (Focus on Pose)
- Silhouette Masking (Ignore Texture)
- Auto-Cropping (Ignore Scale/Position)
- Feature Whitening (Maximize Score Separation)
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel

# --- CONFIGURATION ---
TARGET_IMAGE_PATH = "sketches/chairs/1.png"
CANDIDATES_FOLDER = "sketches/chairs/expanded_36/0/"
RESULT_PATH = "match_result.png"

# 1. LAYER SELECTION
# Layer 9 is usually the best balance for structural pose
LAYER_INDEX = 9 

# 2. MASKING DETAILS
USE_SILHOUETTE = True

# ---------------------------------------------------------------------

def load_dinov2():
    print(f"Loading DINOv2-base (Layer {LAYER_INDEX})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    return processor, model, device

def preprocess_structure(image_path):
    """
    1. Load Image
    2. Convert to Silhouette (Black Ink / White Bg)
    3. Auto-Crop to the content (Removes empty whitespace)
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found {image_path}")
        return None
        
    img = Image.open(image_path).convert("RGB")
    
    if USE_SILHOUETTE:
        arr = np.array(img)
        # Threshold to find ink
        is_ink = arr.mean(axis=2) < 240
        
        # Create fresh canvas
        silhouette = np.ones_like(arr) * 255
        silhouette[is_ink] = [0, 0, 0]
        img = Image.fromarray(silhouette)

    # --- AUTO-CROP LOGIC ---
    # Invert so ink is white (getbbox looks for non-zero pixels)
    inverted = ImageOps.invert(img)
    bbox = inverted.getbbox()
    
    if bbox:
        # Crop to the drawing
        img = img.crop(bbox)
        
        # Optional: Add a small white border back so it's not cramped
        # DINO likes a little context
        w, h = img.size
        pad = max(w, h) // 10
        new_size = (w + 2*pad, h + 2*pad)
        padded = Image.new("RGB", new_size, (255, 255, 255))
        padded.paste(img, (pad, pad))
        img = padded

    return img

def get_raw_embedding(image_path, processor, model, device):
    """Returns the un-normalized embedding vector."""
    image = preprocess_structure(image_path)
    if image is None: return None
    
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    target_layer = outputs.hidden_states[LAYER_INDEX]
    embedding = target_layer[:, 0, :] # [CLS] token
    return embedding

def create_comparison_image(target_path, best_match_path, score):
    img_a = Image.open(target_path).convert("RGB")
    img_b = Image.open(best_match_path).convert("RGB")
    
    ratio = img_a.height / img_b.height
    new_size = (int(img_b.width * ratio), img_a.height)
    img_b = img_b.resize(new_size, Image.Resampling.LANCZOS)
    
    dst = Image.new('RGB', (img_a.width + img_b.width + 10, img_a.height), (255, 255, 255))
    dst.paste(img_a, (0, 0))
    dst.paste(img_b, (img_a.width + 10, 0))
    return dst

def main():
    processor, model, device = load_dinov2()
    
    # 1. Collect Raw Vectors
    print("Collecting embeddings...")
    
    # Target
    target_raw = get_raw_embedding(TARGET_IMAGE_PATH, processor, model, device)
    if target_raw is None: return

    # Candidates
    if not os.path.exists(CANDIDATES_FOLDER):
        return

    candidate_files = sorted([f for f in os.listdir(CANDIDATES_FOLDER) if f.endswith(".png")])
    candidate_raws = []
    valid_filenames = []

    for fname in candidate_files:
        path = os.path.join(CANDIDATES_FOLDER, fname)
        emb = get_raw_embedding(path, processor, model, device)
        if emb is not None:
            candidate_raws.append(emb)
            valid_filenames.append(fname)
            
    if not candidate_raws:
        print("No candidates found.")
        return

    # Stack into a tensor: Shape (N, 768)
    candidates_tensor = torch.cat(candidate_raws, dim=0)
    
    # --- WHITENING TRICK ---
    # Calculate the "Average Chair" vector
    mean_vector = torch.mean(candidates_tensor, dim=0, keepdim=True)
    
    # Subtract mean from everything
    # This removes generic "chair features" and isolates specific "pose features"
    target_centered = target_raw - mean_vector
    candidates_centered = candidates_tensor - mean_vector
    
    # Normalize AFTER centering
    target_norm = F.normalize(target_centered, p=2, dim=1)
    candidates_norm = F.normalize(candidates_centered, p=2, dim=1)
    
    # --- COMPARE ---
    # Dot product of normalized centered vectors
    scores = torch.mm(target_norm, candidates_norm.transpose(0, 1)).squeeze()
    
    best_score = -999.0
    best_idx = -1
    
    print("\n--- Scores (Whitened) ---")
    for i, score_tensor in enumerate(scores):
        val = score_tensor.item()
        fname = valid_filenames[i]
        print(f"  {fname}: {val:.4f}")
        
        if val > best_score:
            best_score = val
            best_idx = i

    best_filename = valid_filenames[best_idx]

    print("\n" + "="*40)
    print(f"WINNER: {best_filename}")
    print(f"Score:  {best_score:.4f}")
    print("="*40)
    
    best_path = os.path.join(CANDIDATES_FOLDER, best_filename)
    comparison = create_comparison_image(TARGET_IMAGE_PATH, best_path, best_score)
    comparison.save(RESULT_PATH)
    print(f"Visual saved to: {RESULT_PATH}")

if __name__ == "__main__":
    main()