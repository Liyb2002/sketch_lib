#!/usr/bin/env python3
"""
match_sequence.py — Cannolize a sequence of sketches to a Master Pose.
Logic:
1. Master Pose = 0.png
2. For every other sketch (1.png, 2.png...):
   - Look at its OWN expansion (expanded_36/1/, expanded_36/2/...)
   - Find the view that matches the structure/pose of 0.png.
   - Save that view to cannolized/{id}.png.
"""

import os
import shutil
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel

# --- CONFIGURATION ---
BASE_DIR = Path("sketches/chairs")
EXPANDED_DIR = BASE_DIR / "expanded_36"
OUTPUT_DIR = BASE_DIR / "cannolized"

# The Master Pose Image
ANCHOR_POSE_FILENAME = "0.png"

# 1. LAYER SELECTION (Layer 9 is best for structural pose)
LAYER_INDEX = 9 

# 2. MASKING DETAILS
USE_SILHOUETTE = True

# ---------------------------------------------------------------------

def load_dinov2():
    print(f"[DINO] Loading DINOv2-base (Layer {LAYER_INDEX})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    return processor, model, device

def preprocess_structure(image_path):
    """
    1. Load Image
    2. Convert to Silhouette (Black Ink / White Bg)
    3. Auto-Crop to the content
    """
    if not os.path.exists(image_path):
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
    inverted = ImageOps.invert(img)
    bbox = inverted.getbbox()
    
    if bbox:
        img = img.crop(bbox)
        # Add padding back so DINO has context
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

def find_best_match(target_path, candidates_folder, processor, model, device):
    """
    Compares target_path (Master Pose) against all images in candidates_folder.
    Returns: (best_image_filename, best_score)
    """
    # 1. Get Target Vector (Master Pose)
    target_raw = get_raw_embedding(target_path, processor, model, device)
    if target_raw is None: 
        print(f"  ! Could not process target: {target_path}")
        return None, 0.0

    # 2. Get Candidate Vectors (The expanded views of the current object)
    if not candidates_folder.exists():
        print(f"  ! Candidate folder missing: {candidates_folder}")
        return None, 0.0

    candidate_files = sorted([f for f in candidates_folder.iterdir() if f.suffix.lower() == ".png"])
    if not candidate_files:
        print("  ! No candidates found.")
        return None, 0.0

    candidate_raws = []
    valid_files = []

    for f in candidate_files:
        emb = get_raw_embedding(f, processor, model, device)
        if emb is not None:
            candidate_raws.append(emb)
            valid_files.append(f)
            
    if not candidate_raws:
        return None, 0.0

    # Stack into tensor: (N, 768)
    candidates_tensor = torch.cat(candidate_raws, dim=0)
    
    # --- WHITENING TRICK ---
    mean_vector = torch.mean(candidates_tensor, dim=0, keepdim=True)
    
    target_centered = target_raw - mean_vector
    candidates_centered = candidates_tensor - mean_vector
    
    target_norm = F.normalize(target_centered, p=2, dim=1)
    candidates_norm = F.normalize(candidates_centered, p=2, dim=1)
    
    # Compare
    scores = torch.mm(target_norm, candidates_norm.transpose(0, 1)).squeeze()
    
    if scores.dim() == 0:
        best_idx = 0
        best_score = scores.item()
    else:
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        
    return valid_files[best_idx], best_score


def main():
    # 1. Setup
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    processor, model, device = load_dinov2()

    # 2. Identify the Master Pose
    anchor_path = BASE_DIR / ANCHOR_POSE_FILENAME
    if not anchor_path.exists():
        print(f"Error: Master pose image {ANCHOR_POSE_FILENAME} not found.")
        return
    
    # 3. Process Sequence
    all_files = sorted([f for f in BASE_DIR.iterdir() if f.suffix == ".png" and f.stem.isdigit()], key=lambda x: int(x.stem))
    
    print(f"\n[Seq] Processing {len(all_files)} sketches. Master Pose: {ANCHOR_POSE_FILENAME}")

    for current_sketch_path in all_files:
        sketch_id = current_sketch_path.stem
        
        # --- SPECIAL CASE: THE MASTER ITSELF ---
        if current_sketch_path.name == ANCHOR_POSE_FILENAME:
            print(f"Processing {current_sketch_path.name} -> Copying Master")
            shutil.copy(current_sketch_path, OUTPUT_DIR / current_sketch_path.name)
            continue

        # --- THE FOLLOWERS ---
        # Goal: Find the view of THIS sketch that matches the MASTER'S pose
        
        candidates_dir = EXPANDED_DIR / sketch_id
        
        print(f"Processing {current_sketch_path.name}...")
        print(f"  > Target Pose: {ANCHOR_POSE_FILENAME}")
        print(f"  > Candidates:  expanded_36/{sketch_id}/")

        best_match, score = find_best_match(anchor_path, candidates_dir, processor, model, device)
        
        if best_match:
            print(f"  > Match Found: {best_match.name} (Score: {score:.4f})")
            shutil.copy(best_match, OUTPUT_DIR / current_sketch_path.name)
        else:
            print(f"  > No match found for {sketch_id}")

    print(f"\n[Seq] Done. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()