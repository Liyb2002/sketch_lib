#!/usr/bin/env python3
"""
extract_components.py (Count-Aware Version with Status Reporting + Instances + Synonyms)

Changes:
1) views_realistic has instance subfolders: sketch/views_realistic/0, /1, /2, ...
   Each contains view_0.png, view_1.png, ... (same sketch, different instances)
   Output mirrors this:
     sketch/segmentation/0/view_0/
     sketch/segmentation/0/view_1/
     ...
     sketch/segmentation/1/view_0/
     ...

2) components_inventory.json includes "synonyms" per component.
   We search using all synonyms (and name) as prompts, but we still SAVE under canonical
   component name as {name}_0, {name}_1, ... (slugified).
   Synonyms are only used for the search mechanism.

Other behavior preserved:
- per-view output folder is cleaned each run (for that instance+view)
- saves crop, mask, overlay per detection
- saves all_components_bbox.png and all_components_bbox.json per view folder
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

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

def _safe_int(x: Any, default: int = 1) -> int:
    try:
        return int(x)
    except Exception:
        return default

def load_inventory_components() -> List[Dict[str, Any]]:
    """
    Reads sketch/program/components_inventory.json

    Returns list of components:
      [
        {
          "name": "Wheel",
          "name_slug": "wheel",
          "count": 2,
          "prompts": ["wheel", "tyre", ...]   # used for search
        },
        ...
      ]
    """
    json_path = PROGRAM_DIR / "components_inventory.json"
    if not json_path.exists():
        print(f"[ERROR] Inventory file not found at: {json_path}")
        return []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print("[ERROR] components_inventory.json must be a JSON object at top-level.")
            return []

        components = data.get("components", [])
        if not isinstance(components, list):
            print("[ERROR] components_inventory.json: 'components' must be a list.")
            return []

        out: List[Dict[str, Any]] = []
        for c in components:
            if not isinstance(c, dict):
                continue
            name = (c.get("name", "") or "").strip()
            if not name:
                continue
            count = _safe_int(c.get("count", 1), default=1)
            syns = c.get("synonyms", [])
            prompts: List[str] = []

            # include name itself
            prompts.append(name)

            # include synonyms if present
            if isinstance(syns, list):
                for s in syns:
                    if isinstance(s, str) and s.strip():
                        prompts.append(s.strip())

            # normalize prompts (lowercase, unique, preserve order)
            seen = set()
            norm_prompts = []
            for p in prompts:
                pl = p.strip().lower()
                if not pl or pl in seen:
                    continue
                seen.add(pl)
                norm_prompts.append(pl)

            out.append({
                "name": name,
                "name_slug": slugify(name),
                "count": max(1, int(count)),
                "prompts": norm_prompts,
            })

        print("[INFO] Loaded inventory components:")
        for comp in out:
            print(f"  - {comp['name']} (count={comp['count']}) prompts={comp['prompts']}")
        return out

    except Exception as e:
        print(f"[ERROR] Failed to load inventory: {e}")
        return []


def _dedupe_mask_candidates(candidates: List[Dict[str, Any]], iou_thresh: float = 0.92) -> List[Dict[str, Any]]:
    """
    Dedupe near-identical masks so synonyms don't produce duplicates.
    Simple greedy: keep higher score, drop if IoU with kept > thresh.
    """
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter) / float(union + 1e-9)

    kept: List[Dict[str, Any]] = []
    for cand in candidates:
        mb = cand["mask_bin"]
        drop = False
        for k in kept:
            if iou(mb, k["mask_bin"]) >= iou_thresh:
                drop = True
                break
        if not drop:
            kept.append(cand)
    return kept


def save_filtered_components(
    label_canonical: str,
    label_slug: str,
    limit: int,
    image: Image.Image,
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    out_dir: Path,
    prompt_used: str,
) -> List[Dict[str, Any]]:
    """
    Saves:
      {label_slug}_{idx}.png
      {label_slug}_{idx}_mask.png
      {label_slug}_{idx}_overlay.png   (overlay is generated later in process_view)
    Returns a list of dict results with mask_bin and box coords.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = image.size
    img_area = W * H

    masks_np = masks.detach().cpu().numpy()
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()

    valid_candidates = []
    for mask, box, score in zip(masks_np, boxes_np, scores_np):
        if float(score) < SCORE_THRESH:
            continue

        mask_bin = mask > 0.5
        if int(mask_bin.sum()) < img_area * MIN_AREA_PCT:
            continue

        valid_candidates.append({
            "mask_bin": mask_bin,
            "box": box,
            "score": float(score),
            "prompt": prompt_used,
        })

    valid_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Greedy dedupe within a single prompt call (usually unnecessary, but safe)
    valid_candidates = _dedupe_mask_candidates(valid_candidates, iou_thresh=0.98)

    top_candidates = valid_candidates[:limit]

    results = []
    for idx, cand in enumerate(top_candidates):
        mask_bin = cand["mask_bin"]
        x0, y0, x1, y1 = map(float, cand["box"])

        results.append({
            "label": label_slug,              # canonical slug used for filenames
            "label_canonical": label_canonical,
            "score": cand["score"],
            "mask_bin": mask_bin,
            "box": (x0, y0, x1, y1),
            "index": idx,                     # per-label final index (will be reindexed later)
            "prompt": cand.get("prompt", ""),
        })

        # crop saving (expanded bbox)
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
    inventory_components: List[Dict[str, Any]],
    processor: Sam3Processor,
    instance_id: str,
):
    print(f"\n--- Processing Instance {instance_id} / View: {img_path.name} ---")

    view_name = img_path.stem  # e.g. "view_0"
    # output: sketch/segmentation/<instance_id>/<view_name>/
    view_out_dir = SEG_OUTPUT_DIR / instance_id / view_name

    # Clean run: remove old outputs for this instance+view
    if view_out_dir.exists():
        import shutil
        shutil.rmtree(view_out_dir)
    view_out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(img_path).convert("RGB")
    W, H = image.size
    state = processor.set_image(image)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
    ]

    all_results: List[Dict[str, Any]] = []
    found_labels: List[str] = []
    missing_labels: List[str] = []

    for comp in inventory_components:
        canonical_name = comp["name"]
        canonical_slug = comp["name_slug"]
        count_limit = int(comp["count"])
        prompts: List[str] = comp["prompts"]

        # Collect candidates across ALL prompts (name + synonyms), then select top N.
        pooled: List[Dict[str, Any]] = []

        for p in prompts:
            try:
                output = processor.set_text_prompt(state=state, prompt=p)
            except Exception as e:
                print(f"  [WARN] Prompt failed for '{canonical_name}' with prompt '{p}': {e}")
                continue

            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

            # Temporarily save results from this prompt into memory (and save crops/masks too),
            # BUT we'll reindex later after pooling and selecting top N.
            # To avoid writing too many files with wrong indices, we DO NOT save yet here.
            # Instead, we convert to candidates, then save only selected ones.

            masks_np = masks.detach().cpu().numpy()
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np[:, 0]
            boxes_np = boxes.detach().cpu().numpy()
            scores_np = scores.detach().cpu().numpy()

            img_area = W * H
            for mask, box, score in zip(masks_np, boxes_np, scores_np):
                sc = float(score)
                if sc < SCORE_THRESH:
                    continue
                mask_bin = mask > 0.5
                if int(mask_bin.sum()) < img_area * MIN_AREA_PCT:
                    continue
                pooled.append({
                    "mask_bin": mask_bin,
                    "box": box,
                    "score": sc,
                    "prompt": p,
                })

        # Sort & dedupe pooled candidates so synonyms don't give duplicates
        pooled.sort(key=lambda x: x["score"], reverse=True)
        pooled = _dedupe_mask_candidates(pooled, iou_thresh=0.92)

        top = pooled[:count_limit]

        if top:
            found_labels.append(f"{canonical_name} ({len(top)}/{count_limit})")

            # Now save ONLY selected detections, with canonical naming: {name_slug}_{idx}
            saved_for_label: List[Dict[str, Any]] = []
            for idx, cand in enumerate(top):
                mask_bin = cand["mask_bin"]
                x0, y0, x1, y1 = map(float, cand["box"])

                r = {
                    "label": canonical_slug,
                    "label_canonical": canonical_name,
                    "score": float(cand["score"]),
                    "mask_bin": mask_bin,
                    "box": (float(x0), float(y0), float(x1), float(y1)),
                    "index": int(idx),
                    "prompt": cand.get("prompt", ""),
                }
                saved_for_label.append(r)

                # --- Save crop + mask for this detection (same logic as before) ---
                cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                w, h = (x1 - x0) * BBOX_SCALE, (y1 - y0) * BBOX_SCALE
                nx0, ny0 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
                nx1, ny1 = min(W, int(cx + w / 2)), min(H, int(cy + h / 2))

                # crop image (masked on white)
                if nx1 > nx0 and ny1 > ny0:
                    crop_img = image.crop((nx0, ny0, nx1, ny1))
                    crop_mask = mask_bin[ny0:ny1, nx0:nx1]
                    if crop_mask.sum() > 0:
                        comp_img = Image.new("RGB", crop_img.size, (255, 255, 255))
                        alpha = Image.fromarray((crop_mask.astype(np.uint8) * 255))
                        comp_img.paste(crop_img, (0, 0), alpha)
                        comp_img.save(view_out_dir / f"{canonical_slug}_{idx}.png")

                # full-res mask
                mask_img = Image.fromarray((mask_bin.astype(np.uint8) * 255), mode="L")
                mask_img.save(view_out_dir / f"{canonical_slug}_{idx}_mask.png")

            # Generate overlays for saved detections
            for r in saved_for_label:
                overlay = image.copy().convert("RGBA")
                mask = r["mask_bin"]
                color = colors[r["index"] % len(colors)]
                color_layer = Image.new("RGBA", overlay.size, color + (0,))
                alpha_mask = Image.fromarray((mask.astype(np.uint8) * 100))
                color_layer.putalpha(alpha_mask)
                overlay.alpha_composite(color_layer)
                overlay.save(view_out_dir / f"{r['label']}_{r['index']}_overlay.png")

            all_results.extend(saved_for_label)
        else:
            missing_labels.append(canonical_name)

    # Status printing
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
            draw.text((x0, y0), text, fill=(255, 255, 255))

        bbox_img.save(view_out_dir / "all_components_bbox.png")

        # ---------------- Save bbox JSON (label matches filename stem) ----------------
        bbox_records = []
        for r in all_results:
            x0, y0, x1, y1 = r["box"]
            stem = f"{r['label']}_{int(r['index'])}"  # matches saved files

            bbox_records.append({
                "label": stem,  # e.g. "wheel_0"
                "canonical_name": r.get("label_canonical", ""),
                "prompt_used": r.get("prompt", ""),
                "score": float(r["score"]),
                "box_xyxy": [float(x0), float(y0), float(x1), float(y1)],
                "box_xywh": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "mask_path": f"{stem}_mask.png",
                "overlay_path": f"{stem}_overlay.png",
                "crop_path": f"{stem}.png",
            })

        with open(view_out_dir / "all_components_bbox.json", "w") as f:
            json.dump({
                "instance": instance_id,
                "view": view_name,
                "image": str(img_path),
                "image_size": {"width": int(W), "height": int(H)},
                "score_thresh": float(SCORE_THRESH),
                "min_area_pct": float(MIN_AREA_PCT),
                "bbox_scale": float(BBOX_SCALE),
                "detections": bbox_records,
            }, f, indent=2)


def _list_instance_dirs(root: Path) -> List[Path]:
    """
    Instance folders are immediate subdirs under views_realistic.
    Example: sketch/views_realistic/0, /1, /2
    If none exist, fallback to treating views_realistic itself as one instance "default".
    """
    if not root.exists():
        return []

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if subdirs:
        return subdirs
    return [root]


def main():
    if not VIEWS_DIR.exists():
        raise SystemExit(f"Views directory not found: {VIEWS_DIR}")

    inventory_components = load_inventory_components()
    if not inventory_components:
        raise SystemExit("No inventory found. Run Step 1 (Inference) first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM3] Loading model on {device}...")

    model = build_sam3_image_model()
    model.to(device)
    model.eval()
    processor = Sam3Processor(model)

    instance_dirs = _list_instance_dirs(VIEWS_DIR)
    if not instance_dirs:
        raise SystemExit(f"No instance folders found under: {VIEWS_DIR}")

    # If VIEWS_DIR itself is used as an instance, name it "default"
    for inst_dir in instance_dirs:
        instance_id = inst_dir.name if inst_dir != VIEWS_DIR else "default"

        image_paths = sorted([
            p for p in inst_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])

        if not image_paths:
            print(f"[WARN] No images found in instance dir: {inst_dir}")
            continue

        for img_path in image_paths:
            process_view(img_path, inventory_components, processor, instance_id)

    print("\n[DONE] Segmentation complete.")


if __name__ == "__main__":
    main()
