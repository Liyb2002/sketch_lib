#!/usr/bin/env python3
"""
extract_components.py (Count-Aware Version with Status Reporting)
1. Reads inventory counts from sketch/program/components_inventory.json
2. For each label, keeps only the Top-N highest scoring detections.
3. Prints found/missing labels for each view.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch

# --- SAM3 IMPORTS ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ---------------- config ----------------

BASE_PATH = Path("sketch")
VIEWS_DIR = BASE_PATH / "views_realistic"
PROGRAM_DIR = BASE_PATH / "program"
SEG_OUTPUT_DIR = BASE_PATH / "segmentation"

SCORE_THRESH = 0.05 
MIN_AREA_PCT = 0.001
BBOX_SCALE   = 1.1


# ---------------- helpers ----------------

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "part"

def load_inventory() -> Dict[str, int]:
    json_path = PROGRAM_DIR / "components_inventory.json"
    if not json_path.exists():
        print(f"[ERROR] Inventory file not found at: {json_path}")
        return {}
        
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            
        components = data.get("components", [])
        inventory = {}
        
        for c in components:
            name = c.get("name", "").strip()
            count = c.get("count", 1)
            if name:
                clean_name = name.lower()
                if clean_name in inventory:
                    inventory[clean_name] = max(inventory[clean_name], int(count))
                else:
                    inventory[clean_name] = int(count)

        print(f"[INFO] Loaded inventory limits: {inventory}")
        return inventory
    except Exception as e:
        print(f"[ERROR] Failed to load inventory: {e}")
        return {}

def save_filtered_components(
    label: str,
    limit: int,
    image: Image.Image,
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    out_dir: Path,
) -> List[Dict[str, Any]]:

    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = image.size
    img_area = W * H
    label_slug = slugify(label)

    masks_np = masks.detach().cpu().numpy()
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()

    valid_candidates = []
    for i, (mask, box, score) in enumerate(zip(masks_np, boxes_np, scores_np)):
        if score < SCORE_THRESH:
            continue
        
        mask_bin = mask > 0.5
        if int(mask_bin.sum()) < img_area * MIN_AREA_PCT:
            continue

        valid_candidates.append({
            "mask_bin": mask_bin,
            "box": box,
            "score": float(score)
        })

    valid_candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = valid_candidates[:limit]

    results = []
    for idx, cand in enumerate(top_candidates):
        mask_bin = cand["mask_bin"]
        x0, y0, x1, y1 = map(float, cand["box"])
        
        results.append({
            "label": label_slug,
            "score": cand["score"],
            "mask_bin": mask_bin,
            "box": (x0, y0, x1, y1),
            "index": idx,
        })

        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        w, h = (x1 - x0) * BBOX_SCALE, (y1 - y0) * BBOX_SCALE
        
        nx0, ny0 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
        nx1, ny1 = min(W, int(cx + w / 2)), min(H, int(cy + h / 2))

        if nx1 > nx0 and ny1 > ny0:
            crop_img = image.crop((nx0, ny0, nx1, ny1))
            crop_mask = mask_bin[ny0:ny1, nx0:nx1]

            if crop_mask.sum() > 0:
                comp = Image.new("RGB", crop_img.size, (255, 255, 255))
                alpha = Image.fromarray((crop_mask.astype(np.uint8) * 255))
                comp.paste(crop_img, (0, 0), alpha)
                comp.save(out_dir / f"{label_slug}_{idx}.png")

        mask_img = Image.fromarray((mask_bin.astype(np.uint8) * 255), mode="L")
        mask_img.save(out_dir / f"{label_slug}_{idx}_mask.png")

    return results


def process_view(
    img_path: Path,
    inventory: Dict[str, int],
    processor: Sam3Processor,
):
    print(f"\n--- Processing View: {img_path.name} ---")
    
    view_name = img_path.stem 
    view_out_dir = SEG_OUTPUT_DIR / view_name
    
    if view_out_dir.exists():
        import shutil
        shutil.rmtree(view_out_dir)
    view_out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(img_path).convert("RGB")
    state = processor.set_image(image)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
    ]

    all_results = []
    found_labels = []
    missing_labels = []

    for label, count_limit in inventory.items():
        output = processor.set_text_prompt(state=state, prompt=label)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        
        # Pass to filter
        results = save_filtered_components(
            label=label,
            limit=count_limit,
            image=image,
            masks=masks,
            boxes=boxes,
            scores=scores,
            out_dir=view_out_dir
        )

        if results:
            found_labels.append(f"{label} ({len(results)}/{count_limit})")
            all_results.extend(results)
            
            # Generate Overlays
            for r in results:
                overlay = image.copy().convert("RGBA")
                mask = r["mask_bin"]
                color = colors[r["index"] % len(colors)]
                color_layer = Image.new("RGBA", overlay.size, color + (0,))
                alpha_mask = Image.fromarray((mask.astype(np.uint8) * 100))
                color_layer.putalpha(alpha_mask)
                overlay.alpha_composite(color_layer)
                overlay.save(view_out_dir / f"{r['label']}_{r['index']}_overlay.png")
        else:
            missing_labels.append(label)

    # Status Printing
    print(f"  [FOUND]:   {', '.join(found_labels) if found_labels else 'None'}")
    print(f"  [MISSING]: {', '.join(missing_labels) if missing_labels else 'None'}")

    if all_results:
        bbox_img = image.copy()
        draw = ImageDraw.Draw(bbox_img)
        for i, r in enumerate(all_results):
            x0, y0, x1, y1 = r["box"]
            col = colors[i % len(colors)]
            draw.rectangle([x0, y0, x1, y1], outline=col, width=3)
            text = f"{r['label']}_{r['index']}"
            bbox = draw.textbbox((x0, y0), text)
            draw.rectangle(bbox, fill=col)
            draw.text((x0, y0), text, fill=(255,255,255))
        bbox_img.save(view_out_dir / "all_components_bbox.png")


def main():
    if not VIEWS_DIR.exists():
        raise SystemExit(f"Views directory not found: {VIEWS_DIR}")
    
    inventory = load_inventory()
    if not inventory:
        raise SystemExit("No inventory found. Run Step 1 (Inference) first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM3] Loading model on {device}...")
    
    model = build_sam3_image_model()
    model.to(device)
    model.eval()
    processor = Sam3Processor(model)

    image_paths = sorted([
        p for p in VIEWS_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])

    for img_path in image_paths:
        process_view(img_path, inventory, processor)

    print("\n[DONE] Segmentation complete.")


if __name__ == "__main__":
    main()