import os
import torch
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image


def _normalize_name_for_merge(name: str) -> str:
    """Return a merge key based on the last meaningful word."""
    if not isinstance(name, str):
        return ""

    s = name.strip().lower()

    # Remove simple punctuation and normalize whitespace
    for ch in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", '"', "'"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())

    if not s:
        return ""

    words = s.split(" ")
    last = words[-1]

    # Optional tiny singularization: wheels -> wheel
    if last.endswith("s") and len(last) > 3:
        last = last[:-1]

    return last


def merge_components_by_last_word(parsed: dict) -> dict:
    """
    Merge components with names sharing the same last word.
    Example: "Front Wheel" + "Back Wheel" -> "Wheel" (count summed)
    """
    if not isinstance(parsed, dict):
        return parsed

    comps = parsed.get("components", [])
    if not isinstance(comps, list):
        return parsed

    merged = {}
    order = []  # preserve first-seen order of merge keys

    for comp in comps:
        if not isinstance(comp, dict):
            continue

        name = comp.get("name", "")
        key = _normalize_name_for_merge(name)

        # If name is unusable, keep it isolated
        if not key:
            key = f"__raw__:{name}"

        # Robust count parse
        count = comp.get("count", 1)
        try:
            count = int(count)
        except Exception:
            count = 1
        count = max(count, 0)

        if key not in merged:
            merged[key] = {
                "count": count,
                "name_candidates": [name] if isinstance(name, str) and name.strip() else [],
            }
            order.append(key)
        else:
            merged[key]["count"] += count
            if isinstance(name, str) and name.strip():
                merged[key]["name_candidates"].append(name)

    # Build final components list with refreshed component_ids
    new_components = []
    for i, key in enumerate(order, start=1):
        if key.startswith("__raw__:"):
            canonical_name = key.split(":", 1)[1].strip() or "Component"
        else:
            canonical_name = key.title()

        new_components.append(
            {
                "component_id": f"C{i:02d}",
                "name": canonical_name,
                "count": merged[key]["count"],
            }
        )

    parsed["components"] = new_components
    return parsed


def run_inference(base_path="."):
    print("\n" + "=" * 40)
    print("STEP 1: Structural Decomposition (Non-Overlapping)")
    print("=" * 40)

    # --- PATH SETUP ---
    output_dir = os.path.join(base_path, "sketch", "program")
    os.makedirs(output_dir, exist_ok=True)

    master_path = os.path.join(base_path, "sketch", "input.png")
    views_path = os.path.join(base_path, "sketch", "views")
    output_file = os.path.join(output_dir, "components_inventory.json")

    # --- LOAD MODEL ---
    model_path = "Qwen/Qwen2-VL-7B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # --- LOAD IMAGES ---
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Master sketch not found at {master_path}")

    # Prioritize the input.png (Master Sketch) as the primary source of truth
    view_paths = []
    if os.path.isdir(views_path):
        view_paths = sorted(
            [
                os.path.join(views_path, f)
                for f in os.listdir(views_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )[:6]

    images = [Image.open(master_path)] + [Image.open(p) for p in view_paths]

    # --- UPDATED PROMPT: TASK DECOMPOSITION ---
    system_prompt = """
    You are a Design. Your task is to perform a Structural Decomposition of the object shown in the Sketch drawing.

    TASK: Decompose the assembly into a set of discrete, non-overlapping components. You should have more than 5 but less than 15 components.

    CONSTRAINTS:
    1. **Exclusivity (No Overlaps):** Do not list a whole assembly and its sub-parts as separate items.
    2. **Completeness:** The sum of these components should equal the entire physical volume of the object in the image.
    3. **Functional Naming:** Use technical terminology to describe the role of each component.

    Output strictly in JSON format:
    {
      "assembly_name": "Name of the overall object",
      "decomposition_strategy": "Brief explanation of how you divided the parts without overlap",
      "components": [
        {
          "component_id": "C01",
          "name": "Technical Name",
          "count": 1
        }
      ]
    }
    """

    # --- INFERENCE ---
    content_list = [{"type": "image", "image": img} for img in images]
    content_list.append({"type": "text", "text": system_prompt})

    messages = [{"role": "user", "content": content_list}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(device)

    print("Executing decomposition task...")
    generated_ids = model.generate(**inputs, max_new_tokens=1536)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    # --- CLEAN & SAVE ---
    clean_json = output_text.replace("```json", "").replace("```", "").strip()
    if "{" in clean_json:
        start = clean_json.find("{")
        end = clean_json.rfind("}") + 1
        clean_json = clean_json[start:end]

    try:
        parsed = json.loads(clean_json)

        # --- FINAL EDIT: MERGE SEMANTICALLY SIMILAR COMPONENTS (BY LAST WORD) ---
        parsed = merge_components_by_last_word(parsed)

        with open(output_file, "w") as f:
            json.dump(parsed, f, indent=2)
        print(f"Success! Decomposition saved to: {output_file}")

    except json.JSONDecodeError:
        with open(output_file.replace(".json", ".txt"), "w") as f:
            f.write(clean_json)
        print(f"[WARN] Model output was not valid JSON. Saved raw text to: {output_file.replace('.json', '.txt')}")
