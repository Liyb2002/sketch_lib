#!/usr/bin/env python
"""
discover_parts_with_vlm.py

Use a local VLM (Qwen2.5-VL-7B-Instruct) to:
- Scan ./sketches/{object_subfolder}/
- Sample a few images from each subfolder
- Infer a shared object type and 6–9 semantic components/parts
- Save:
    sketches/{object_subfolder}/components.json

JSON format:
{
  "object_type": "office_chair",
  "components": ["seat", "backrest", "armrest", "base", "wheels", ...]
}
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

# ---------------- config ----------------

SKETCHES_ROOT = Path("sketches")  # folder containing chair/, car/, etc.
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_IMAGES_PER_CLASS = 4          # how many images per subfolder to show VLM
MAX_NEW_TOKENS = 256


# --------------- VLM setup ---------------

def load_vlm():
    print(f"Loading VLM: {MODEL_ID}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model loaded on device:", next(model.parameters()).device)
    return model, processor


# --------------- helpers -----------------

def _find_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    imgs = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    return imgs


def _clean_to_json(text: str) -> Dict[str, Any]:
    """
    Try to extract a JSON object from the model output.
    Accepts raw JSON or ```json ... ``` fenced blocks.
    """
    # Grab ```json ... ``` code block if present
    m = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1)
    else:
        candidate = text

    # Strip junk before/after first/last brace
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start:end + 1]

    return json.loads(candidate)


def _build_messages(images: List[Path]) -> List[Dict[str, Any]]:
    """
    Build a single multi-image chat message for Qwen2.5-VL.
    """
    content: List[Dict[str, Any]] = []

    for img_path in images:
        # Qwen2.5-VL accepts local paths via "file://"
        content.append({
            "type": "image",
            "image": f"file://{img_path.resolve()}"
        })

    # Chain-of-thought style internally but JSON-only output:
    prompt = (
        "You are an expert in product design and part decomposition.\n"
        "You will think step by step internally, but ONLY output JSON.\n\n"
        "These images show different instances of the SAME object category.\n"
        "1) First, infer a concise object type (e.g., 'office_chair', 'bicycle').\n"
        "2) Then, infer BETWEEN 6 AND 9 semantic components/parts that appear in at least one image.\n"
        "   - If there are only a few obvious parts, subdivide them into meaningful sub-parts.\n"
        "   - It is OK for components to be related or overlapping (e.g., 'wheel' and 'tire').\n"
        "3) Use short, generic names (e.g., 'seat', 'backrest', 'wheel', 'handle').\n"
        "4) Use snake_case lowercase names for all components.\n\n"
        "Return ONLY a JSON object with this structure, and nothing else:\n"
        "{\n"
        '  \"object_type\": \"<string>\",\n'
        '  \"components\": [\"<component_1>\", \"<component_2\", \"...\"]\n'
        "}"
    )

    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    return messages


def analyze_folder(folder: Path, model, processor) -> Dict[str, Any]:
    images = _find_images(folder)
    if not images:
        print(f"[skip] No images in {folder}")
        return {}

    images = images[:MAX_IMAGES_PER_CLASS]
    print(f"[info] {folder.name}: using {len(images)} image(s): "
          f"{', '.join(p.name for p in images)}")

    messages = _build_messages(images)

    # Prepare inputs (multi-image) using qwen-vl-utils
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    # Cut off the prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    raw = output_texts[0]
    # print("RAW OUTPUT:", raw)  # uncomment for debugging

    try:
        parsed = _clean_to_json(raw)
    except Exception as e:
        print(f"[warn] Failed to parse JSON for {folder.name}: {e}")
        return {}

    # Light sanity checks
    obj_type = str(parsed.get("object_type", folder.name)).strip()
    comps = parsed.get("components", [])
    if not isinstance(comps, list):
        comps = []

    # normalize and dedupe EXACT duplicates,
    # but allow semantic overlap (e.g., "tire" vs "wheel")
    norm_comps: List[str] = []
    for c in comps:
        s = str(c).strip().lower()
        s = s.replace(" ", "_")
        if not s:
            continue
        if s not in norm_comps:
            norm_comps.append(s)

    # enforce <= 9 components if possible
    if len(norm_comps) > 9:
        norm_comps = norm_comps[:9]

    # if model gave fewer than 6, we keep them but warn
    if len(norm_comps) < 6:
        print(
            f"[warn] {folder.name}: only {len(norm_comps)} components returned; "
            f"prompt asked for 6–9."
        )

    return {
        "object_type": obj_type if obj_type else folder.name,
        "components": norm_comps,
    }


def save_components(folder: Path, data: Dict[str, Any]):
    out_path = folder / "components.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[ok] Saved {out_path}")


# --------------- main --------------------

def main():
    if not SKETCHES_ROOT.exists():
        raise SystemExit(f"{SKETCHES_ROOT} not found")

    model, processor = load_vlm()

    for sub in sorted(SKETCHES_ROOT.iterdir()):
        if not sub.is_dir():
            continue

        result = analyze_folder(sub, model, processor)
        if not result:
            continue
        save_components(sub, result)


if __name__ == "__main__":
    main()
