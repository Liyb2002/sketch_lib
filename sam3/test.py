#!/usr/bin/env python3
"""
test.py — quick SAM3 exploratory script.

- Loads facebook SAM3 via sam3.model_builder
- Segments "objects" in 0.png
- Saves each segmented component as its own PNG on white background
  in ./components/
"""

from pathlib import Path
import os

import torch
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def save_components(image: Image.Image,
                    masks: torch.Tensor,
                    boxes: torch.Tensor,
                    scores: torch.Tensor,
                    out_dir: Path,
                    score_thresh: float = 0.3,
                    bbox_scale: float = 1.2) -> None:
    """
    Save each high-confidence mask as a cropped component with white background.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Move to CPU + numpy for easier handling
    masks_np = masks.detach().cpu().numpy()       # (N, H, W)
    boxes_np = boxes.detach().cpu().numpy()       # (N, 4) [x0, y0, x1, y1]
    scores_np = scores.detach().cpu().numpy()     # (N,)

    W, H = image.size

    kept = 0
    for i, (mask, box, score) in enumerate(zip(masks_np, boxes_np, scores_np)):
        if score < score_thresh:
            continue

        # Binary mask
        mask_bin = mask > 0.5

        # Original box
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)

        # Expand bbox by bbox_scale around its center
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        w = (x1 - x0) * bbox_scale
        h = (y1 - y0) * bbox_scale

        new_x0 = max(0, int(round(cx - w / 2)))
        new_y0 = max(0, int(round(cy - h / 2)))
        new_x1 = min(W, int(round(cx + w / 2)))
        new_y1 = min(H, int(round(cy + h / 2)))

        if new_x1 <= new_x0 or new_y1 <= new_y0:
            continue

        # Crop image and mask to expanded bbox
        crop_img = image.crop((new_x0, new_y0, new_x1, new_y1))
        crop_mask = mask_bin[new_y0:new_y1, new_x0:new_x1]

        if crop_mask.sum() == 0:
            # mask empty in this region; skip
            continue

        # Create white background
        comp = Image.new("RGB", crop_img.size, (255, 255, 255))

        # Mask as 8-bit alpha
        alpha = Image.fromarray((crop_mask.astype(np.uint8) * 255), mode="L")

        # Paste original crop using mask as transparency
        comp.paste(crop_img, (0, 0), alpha)

        out_path = out_dir / f"{kept}.png"
        comp.save(out_path)
        print(f"Saved component {kept} with score={score:.3f} -> {out_path}")
        kept += 1

    if kept == 0:
        print("No components passed the score threshold; nothing saved.")


def main():
    root = Path(__file__).resolve().parent
    img_path = root / "0.png"
    if not img_path.is_file():
        raise FileNotFoundError(f"Could not find image: {img_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading SAM3 image model…")
    model = build_sam3_image_model()
    model.to(device)
    model.eval()

    processor = Sam3Processor(model)

    # Load image
    image = Image.open(img_path).convert("RGB")

    # Set the image in the processor
    print("Running SAM3 on 0.png …")
    state = processor.set_image(image)

    # Simple open-vocab prompt. You can change this to e.g. "chair", "car", etc.
    text_prompt = "object"
    output = processor.set_text_prompt(state=state, prompt=text_prompt)

    masks = output["masks"]   # (N, H, W)
    boxes = output["boxes"]   # (N, 4)
    scores = output["scores"] # (N,)

    print(f"Got {masks.shape[0]} masks from SAM3 for prompt '{text_prompt}'.")

    # Save segmented components
    out_dir = root / "components"
    # Optional: clean old outputs
    if out_dir.exists():
        for p in out_dir.glob("*.png"):
            p.unlink()

    save_components(image, masks, boxes, scores, out_dir)


if __name__ == "__main__":
    main()
