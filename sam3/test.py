#!/usr/bin/env python3
"""
extract_components_with_sam3.py

Pipeline:

sketches/
  {object_type}/
    components.json   # produced by discover_parts_with_vlm.py
    individual_object/
      0/
        view0.png
        view1.png
        ...
      1/
        view0.png
        ...

For each object_type:
  - load components.json -> list of labels
  - for each individual_object/{x}
    - for each view image
      - run SAM3 with each label as a text prompt
      - extract component images (cropped on white background)
      - save to:
          sketches/{object_type}/individual_object/{x}/{view_stem}/
            {label_slug}_{k}.png
"""

from pathlib import Path
import json
import re
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ---------------- config ----------------

SKETCHES_ROOT = Path("sketches")  # root folder with object subfolders

SCORE_THRESH = 0.05   # min SAM3 score to accept a mask
MIN_AREA_PCT = 0.001  # min mask area as fraction of full image
BBOX_SCALE   = 1.2    # expand bounding box for nicer context crop
MAX_INSTANCES_PER_LABEL = 8  # safety cap per view per label


# ---------------- helpers ----------------

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "part"


def find_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def load_components_json(obj_dir: Path) -> List[str]:
    comp_path = obj_dir / "components.json"
    if not comp_path.is_file():
        print(f"[skip] No components.json in {obj_dir}")
        return []
    with open(comp_path, "r") as f:
        data = json.load(f)
    comps = data.get("components", [])
    if not isinstance(comps, list):
        comps = []
    labels = [str(c).strip().lower().replace(" ", "_") for c in comps if str(c).strip()]
    print(f"[info] {obj_dir.name}: loaded {len(labels)} component labels")
    return labels


def save_components_for_label(
    label: str,
    image: Image.Image,
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    out_dir: Path,
) -> int:
    """
    Save cropped component images for one label in one view.
    returns: number of saved instances
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = image.size
    img_area = W * H
    label_slug = slugify(label)

    masks_np = masks.detach().cpu().numpy()      # could be (N, H, W) or (N, 1, H, W)
    # If there's a singleton channel dimension, squeeze it out
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0, :, :]

    boxes_np = boxes.detach().cpu().numpy()      # (N, 4)
    scores_np = scores.detach().cpu().numpy()    # (N,)

    saved = 0
    for i, (mask, box, score) in enumerate(zip(masks_np, boxes_np, scores_np)):
        if score < SCORE_THRESH:
            continue

        mask_bin = mask > 0.5
        area = int(mask_bin.sum())
        if area < img_area * MIN_AREA_PCT:
            continue

        x0, y0, x1, y1 = map(float, box)

        # expand bbox
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        w = (x1 - x0) * BBOX_SCALE
        h = (y1 - y0) * BBOX_SCALE

        new_x0 = max(0, int(round(cx - w / 2)))
        new_y0 = max(0, int(round(cy - h / 2)))
        new_x1 = min(W, int(round(cx + w / 2)))
        new_y1 = min(H, int(round(cy + h / 2)))

        if new_x1 <= new_x0 or new_y1 <= new_y0:
            continue

        crop_img = image.crop((new_x0, new_y0, new_x1, new_y1))
        crop_mask = mask_bin[new_y0:new_y1, new_x0:new_x1]

        # If this is still 3D (e.g. (1, h, w)), squeeze again
        if crop_mask.ndim == 3:
            crop_mask = crop_mask.squeeze()

        if crop_mask.sum() == 0:
            continue

        # white background
        comp = Image.new("RGB", crop_img.size, (255, 255, 255))
        # Pillow can infer L mode from uint8 array; no need for mode="L"
        alpha_arr = (crop_mask.astype(np.uint8) * 255)
        alpha = Image.fromarray(alpha_arr)

        comp.paste(crop_img, (0, 0), alpha)

        out_path = out_dir / f"{label_slug}_{saved}.png"
        comp.save(out_path)
        saved += 1

        if saved >= MAX_INSTANCES_PER_LABEL:
            break

    return saved


# ---------------- main logic ----------------

def process_view_image(
    img_path: Path,
    labels: List[str],
    processor: Sam3Processor,
    model_device: torch.device,
):
    """
    For a single view image:
      - create a folder {view_stem}/ in the same dir
      - for each label, run SAM3 and save component crops
    """
    print(f"  [view] {img_path.name}")
    image = Image.open(img_path).convert("RGB")

    # Where to save:
    view_dir = img_path.with_suffix("")  # e.g. "view0.png" -> "view0"
    view_dir.mkdir(exist_ok=True)

    # Clear previous crops in this view folder
    for p in view_dir.glob("*.png"):
        p.unlink()

    # SAM3: set image once, reuse state for all labels
    state = processor.set_image(image)

    total_saved = 0
    for label in labels:
        output = processor.set_text_prompt(state=state, prompt=label)

        masks = output["masks"]    # (N, H, W)
        boxes = output["boxes"]    # (N, 4)
        scores = output["scores"]  # (N,)

        if masks.shape[0] == 0:
            print(f"    [label '{label}'] no masks returned")
            continue

        n_saved = save_components_for_label(
            label=label,
            image=image,
            masks=masks,
            boxes=boxes,
            scores=scores,
            out_dir=view_dir,
        )
        print(f"    [label '{label}'] saved {n_saved} instances")
        total_saved += n_saved

    if total_saved == 0:
        print("    [info] no components saved for this view (all labels)")


def process_object_instance(
    instance_dir: Path,
    labels: List[str],
    processor: Sam3Processor,
    model_device: torch.device,
):
    print(f"[instance] {instance_dir}")
    images = find_images(instance_dir)
    if not images:
        print(f"  [skip] no images in {instance_dir}")
        return

    for img_path in images:
        process_view_image(
            img_path=img_path,
            labels=labels,
            processor=processor,
            model_device=model_device,
        )


def main():
    if not SKETCHES_ROOT.exists():
        raise SystemExit(f"{SKETCHES_ROOT} not found")

    # Load SAM3 once
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[sam3] using device: {device}")

    print("[sam3] loading image model…")
    model = build_sam3_image_model()
    model.to(device)
    model.eval()
    processor = Sam3Processor(model)
    print("[sam3] loaded.")

    # Walk object types
    for obj_dir in sorted(SKETCHES_ROOT.iterdir()):
        if not obj_dir.is_dir():
            continue

        labels = load_components_json(obj_dir)
        if not labels:
            continue

        inst_root = obj_dir / "individual_object"
        if not inst_root.is_dir():
            print(f"[warn] {obj_dir.name}: no individual_object/ folder, skipping")
            continue

        for instance_dir in sorted(inst_root.iterdir()):
            if not instance_dir.is_dir():
                continue
            process_object_instance(
                instance_dir=instance_dir,
                labels=labels,
                processor=processor,
                model_device=device,
            )

    print("Done.")


if __name__ == "__main__":
    main()
