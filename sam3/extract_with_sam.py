#!/usr/bin/env python3
"""
extract_components_with_sam3.py (extended)
Now also saves:
1) overlays for each mask on the original image
2) a single bounding-box visualization per view
"""

from pathlib import Path
import json
import re
from typing import List, Dict, Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ---------------- config ----------------

SKETCHES_ROOT = Path("sketches")

SCORE_THRESH = 0.05
MIN_AREA_PCT = 0.001
BBOX_SCALE   = 1.2
MAX_INSTANCES_PER_LABEL = 8


# ---------------- helpers ----------------

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "part"


def find_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def load_components_json(obj_dir: Path) -> List[str]:
    comp_path = obj_dir / "components.json"
    if not comp_path.is_file():
        print(f"[skip] No components.json in {obj_dir}")
        return []
    with open(comp_path, "r") as f:
        data = json.load(f)
    comps = data.get("components", [])
    labels = [str(c).strip().lower().replace(" ", "_") for c in comps if str(c).strip()]
    print(f"[info] {obj_dir.name}: loaded {len(labels)} component labels")
    return labels


# NEW: Return mask+box info for overlays + bbox visualization
def save_components_for_label(
    label: str,
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

    results = []
    saved = 0

    for i, (mask, box, score) in enumerate(zip(masks_np, boxes_np, scores_np)):
        if score < SCORE_THRESH:
            continue

        mask_bin = mask > 0.5
        area = int(mask_bin.sum())
        if area < img_area * MIN_AREA_PCT:
            continue

        x0, y0, x1, y1 = map(float, box)

        results.append({
            "label": label_slug,
            "score": float(score),
            "mask_bin": mask_bin,
            "box": (x0, y0, x1, y1),
            "index": saved,
        })

        # Expand bbox
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        w = (x1 - x0) * BBOX_SCALE
        h = (y1 - y0) * BBOX_SCALE

        new_x0 = max(0, int(cx - w / 2))
        new_y0 = max(0, int(cy - h / 2))
        new_x1 = min(W, int(cx + w / 2))
        new_y1 = min(H, int(cy + h / 2))

        crop_img = image.crop((new_x0, new_y0, new_x1, new_y1))
        crop_mask = mask_bin[new_y0:new_y1, new_x0:new_x1]

        if crop_mask.sum() == 0:
            continue

        # White background crop
        comp = Image.new("RGB", crop_img.size, (255, 255, 255))
        alpha = Image.fromarray((crop_mask.astype(np.uint8) * 255))
        comp.paste(crop_img, (0, 0), alpha)

        comp.save(out_dir / f"{label_slug}_{saved}.png")
        saved += 1

        if saved >= MAX_INSTANCES_PER_LABEL:
            break

    return results


def process_view_image(
    img_path: Path,
    labels: List[str],
    processor: Sam3Processor,
    model_device: torch.device,
):
    print(f"  [view] {img_path.name}")
    image = Image.open(img_path).convert("RGB")

    view_dir = img_path.with_suffix("")
    view_dir.mkdir(exist_ok=True)
    for p in view_dir.glob("*.png"):
        p.unlink()

    state = processor.set_image(image)

    colors = [
        (255,0,0), (0,255,0), (0,0,255),
        (255,255,0), (255,0,255), (0,255,255),
        (255,128,0), (128,0,255),
    ]

    all_results = []
    total_saved = 0

    for label in labels:
        output = processor.set_text_prompt(state=state, prompt=label)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        if masks.shape[0] == 0:
            continue

        results = save_components_for_label(label, image, masks, boxes, scores, out_dir=view_dir)
        all_results.extend(results)
        total_saved += len(results)

        # Save overlays per mask
        for r in results:
            overlay = image.copy().convert("RGBA")
            mask = r["mask_bin"]

            color = colors[r["index"] % len(colors)]
            alpha = Image.fromarray((mask.astype(np.uint8) * 120))  # transparency

            color_img = Image.new("RGBA", overlay.size, color + (0,))
            color_img.putalpha(alpha)
            overlay = Image.alpha_composite(overlay, color_img)

            overlay.save(view_dir / f"{r['label']}_{r['index']}_overlay.png")

    # Save combined bbox visualization
    if all_results:
        bbox_img = image.copy()
        draw = ImageDraw.Draw(bbox_img)

        for i, r in enumerate(all_results):
            x0, y0, x1, y1 = r["box"]
            col = colors[i % len(colors)]
            draw.rectangle([x0, y0, x1, y1], outline=col, width=3)
            draw.text((x0, y0), r["label"], fill=col)

        bbox_img.save(view_dir / "all_components_bbox.png")
    else:
        print("    [info] no components saved in this view.")


def process_object_instance(
    instance_dir: Path,
    labels: List[str],
    processor: Sam3Processor,
    model_device: torch.device,
):
    print(f"[instance] {instance_dir}")
    images = find_images(instance_dir)
    for img_path in images:
        process_view_image(img_path, labels, processor, model_device)


def main():
    if not SKETCHES_ROOT.exists():
        raise SystemExit(f"{SKETCHES_ROOT} not found")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[sam3] using device: {device}")

    print("[sam3] loading image model…")
    model = build_sam3_image_model()
    model.to(device)
    model.eval()
    processor = Sam3Processor(model)
    print("[sam3] loaded.")

    for obj_dir in sorted(SKETCHES_ROOT.iterdir()):
        if not obj_dir.is_dir():
            continue

        labels = load_components_json(obj_dir)
        if not labels:
            continue

        inst_root = obj_dir / "individual_object"
        if not inst_root.is_dir():
            continue

        for instance_dir in sorted(inst_root.iterdir()):
            if instance_dir.is_dir():
                process_object_instance(instance_dir, labels, processor, device)

    print("Done.")


if __name__ == "__main__":
    main()
