#!/usr/bin/env python3
# vlm_pipeline/synonym.py

import os
import json
import re
from typing import Dict, Any, List, Optional

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


# ----------------------------
# Helpers
# ----------------------------

def _safe_read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("components_inventory.json must contain a JSON object at top-level.")
    return data


def _safe_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def _strip_code_fences(s: str) -> str:
    s = s.replace("```json", "").replace("```", "").strip()
    # crop to outermost braces if present
    if "{" in s and "}" in s:
        start = s.find("{")
        end = s.rfind("}") + 1
        s = s[start:end]
    return s.strip()


def _normalize_syn(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # remove weird quotes/backticks
    s = s.strip("`'\"")
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = _normalize_syn(x)
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def _fallback_synonyms(name: str) -> List[str]:
    # Very conservative fallback (keeps your pipeline moving even if model output fails)
    n = (name or "").strip()
    if not n:
        return ["Component", "Part"]
    # Simple, generic alternates
    return _dedupe_keep_order([f"{n} Part", f"{n} Component"])


# ----------------------------
# Main: add synonyms and overwrite
# ----------------------------

def add_synonyms_overwrite(
    base_path: str = ".",
    model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
    max_new_tokens: int = 512,
) -> None:
    """
    Reads sketch/program/components_inventory.json
    Adds: component["synonyms"] = ["syn1", "syn2"] for each component
    Overwrites the same file.
    """

    program_dir = os.path.join(base_path, "sketch", "program")
    inv_path = os.path.join(program_dir, "components_inventory.json")

    parsed = _safe_read_json(inv_path)
    comps = parsed.get("components", [])
    if not isinstance(comps, list) or len(comps) == 0:
        raise ValueError("components_inventory.json has no valid 'components' list to process.")

    # Build a compact request payload
    request_items = []
    for c in comps:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("component_id", "")).strip()
        nm = str(c.get("name", "")).strip()
        if cid and nm:
            request_items.append({"component_id": cid, "name": nm})

    if not request_items:
        raise ValueError("No components with both component_id and name found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # Prompt: single shot mapping to avoid per-component calls
    system_prompt = f"""
You are a mechanical design assistant.

TASK:
Given a list of components from a CAD/assembly decomposition, produce EXACTLY TWO synonyms (alternative technical names)
for each component name.

RULES:
- Keep synonyms short (1-4 words).
- Prefer functional/technical alternatives (e.g., "handle" -> "grip", "housing" -> "enclosure").
- Do not invent new components.
- Do not include counts or IDs inside synonyms.
- Avoid positional words (front/back/left/right/top/bottom) unless absolutely necessary.
- Output STRICT JSON only, no markdown.
- Synonyms MUST NOT be identical to the original component name (case-insensitive).

INPUT:
{json.dumps({"components": request_items}, indent=2)}

OUTPUT FORMAT (STRICT):
{{
  "synonyms": {{
    "C01": ["syn1", "syn2"],
    "C02": ["syn1", "syn2"]
  }}
}}
""".strip()

    messages = [{"role": "user", "content": [{"type": "text", "text": system_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    print("Adding synonyms to components_inventory.json ...")
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    clean = _strip_code_fences(output_text)

    mapping: Optional[Dict[str, Any]] = None
    try:
        obj = json.loads(clean)
        mapping = obj.get("synonyms", None) if isinstance(obj, dict) else None
        if not isinstance(mapping, dict):
            mapping = None
    except Exception:
        mapping = None

    # Apply synonyms with robust fallback
    for c in comps:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("component_id", "")).strip()
        nm = str(c.get("name", "")).strip()

        syns = None
        if mapping and cid in mapping and isinstance(mapping[cid], list):
            syns = _dedupe_keep_order([str(x) for x in mapping[cid]])
            # enforce exactly 2
            if len(syns) >= 2:
                syns = syns[:2]
            else:
                # pad with fallback
                fb = _fallback_synonyms(nm)
                syns = _dedupe_keep_order(syns + fb)[:2]

        else:
            syns = _fallback_synonyms(nm)[:2]

        c["synonyms"] = syns

    parsed["components"] = comps
    _safe_write_json(inv_path, parsed)

    print(f"Done. Overwrote: {inv_path}")
