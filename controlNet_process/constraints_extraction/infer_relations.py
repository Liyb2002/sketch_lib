#!/usr/bin/env python3
"""
constraints_extraction/infer_relations.py

LLM-based relation inference from labels ONLY (no image, no rendering).
Works for any object category by conditioning on assembly name + label strings.

Entry point kept the same:
  from constraints_extraction.infer_relations import main as infer_relations_main
  infer_relations_main()

Inputs:
  sketch/dsl_optimize/registry.json
  sketch/components_inventory.json   (to read assembly_name)

Outputs:
  sketch/dsl_optimize/relations.json

Schema (kept compatible with your existing pipeline):
{
  "model": "...",
  "assembly_name": "...",
  "labels": [...],
  "neighboring_pairs": [...],
  "same_pairs": [...],     # NOTE: we reinterpret this as "related/similar components"
  "notes": "...",
  "confidence": 0..1
}
"""

import os
import json
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

SKETCH_ROOT = os.path.join(REPO_ROOT, "sketch")
DSL_DIR = os.path.join(SKETCH_ROOT, "dsl_optimize")

REGISTRY_JSON = os.path.join(DSL_DIR, "registry.json")
COMPONENTS_INVENTORY_JSON = os.path.join(SKETCH_ROOT, "components_inventory.json")
OUT_RELATIONS_JSON = os.path.join(DSL_DIR, "relations.json")

# Text instruct model for label-only reasoning
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _extract_labels_from_registry(registry: Dict[str, Any]) -> List[str]:
    """
    registry.json is built by build_registry_from_cluster_map().
    Typical schema: cluster_id -> {"label": "...", "color": ...}
    Sometimes: cluster_id -> "label"
    """
    labels: List[str] = []
    if isinstance(registry, dict):
        for _, v in registry.items():
            if isinstance(v, dict):
                lab = v.get("label", None)
                if lab is not None:
                    labels.append(str(lab))
            elif isinstance(v, str):
                labels.append(str(v))
    return sorted(set(labels))


def _read_assembly_name() -> str:
    """
    Reads sketch/components_inventory.json and returns assembly_name if present.
    If missing, returns "Unknown".
    """
    if not os.path.exists(COMPONENTS_INVENTORY_JSON):
        return "Unknown"
    try:
        data = _load_json(COMPONENTS_INVENTORY_JSON)
        if isinstance(data, dict) and isinstance(data.get("assembly_name", None), str):
            return data["assembly_name"].strip() or "Unknown"
    except Exception:
        pass
    return "Unknown"


def _robust_parse_json(s: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from a model response.
    """
    s = (s or "").strip()

    # Direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Strip code fences
    if "```" in s:
        parts = s.split("```")
        candidates = [parts[i] for i in range(1, len(parts), 2)]
        for c in candidates:
            c = c.strip()
            if c.lower().startswith("json"):
                c = c[4:].strip()
            try:
                return json.loads(c)
            except Exception:
                continue

    # Find outermost {...}
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        s2 = s[l : r + 1]
        try:
            return json.loads(s2)
        except Exception:
            pass

    return {
        "neighboring_pairs": [],
        "same_pairs": [],
        "notes": f"Failed to parse JSON. Raw: {s[:400]}",
        "confidence": 0.0,
    }


def _load_llm(model_path: str = MODEL_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _call_llm_for_relations(
    model,
    tokenizer,
    assembly_name: str,
    labels: List[str],
    max_new_tokens: int = 700,
    temperature: float = 0.25,
) -> Dict[str, Any]:
    """
    Ask the LLM to infer relations based on label semantics only.
    Returns parsed JSON dict.
    """

    # IMPORTANT:
    # We keep the output key "same_pairs" for compatibility with your main script,
    # but we redefine it as "related/similar components" rather than "exact same".
    prompt_user = f"""
You are given semantic component labels for a single object.

Assembly / object type: "{assembly_name}"

Labels (exact strings):
{labels}

Infer RELATIONS between labels using ONLY the label names and the assembly name.
This is approximate. Be permissive: include plausible relations with lower confidence.

Return ONLY valid JSON with exactly these keys:
{{
  "neighboring_pairs": [{{"a": "<label>", "b": "<label>", "confidence": 0-1, "evidence": "<short>"}} ...],
  "same_pairs": [{{"a": "<label>", "b": "<label>", "confidence": 0-1, "evidence": "<short>"}} ...],
  "notes": "<short>",
  "confidence": 0-1
}}

Definitions:
- neighboring_pairs:
  Two parts likely adjacent / attached / connected / in direct contact in a typical instance of this assembly.
  Include plausible adjacency even if not guaranteed; lower confidence if unsure.

- same_pairs (IMPORTANT: interpret as "related/similar components", NOT exact identical):
  Two labels that are the same kind of component or in the same family, such as:
    - left/right variants (mirror duplicates)
    - front/back variants of the same type (front_leg vs rear_leg)
    - repeated instances (leg_0 vs leg_1)
    - same subsystem family (front_wheel vs rear_wheel are both wheels)
    - naming variants/synonyms (tank vs fuel_tank)
  Do NOT require them to be literally the same physical part.

Rules:
- "a" and "b" must be chosen from the provided labels list and must match exactly.
- Do NOT output duplicate unordered pairs (treat (a,b) same as (b,a)).
- Provide a short evidence string referencing label cues (e.g., "front/back variant", "both wheels", "left/right mirror", "subsystem family").
- Confidence guidance:
    0.85-1.0: very likely (clear family/variant cues)
    0.6-0.85: likely (reasonable family relation)
    0.3-0.6: plausible adjacency guess
- If no relations are supported, output empty lists.
- Output JSON only. No extra text.
""".strip()

    messages = [
        {"role": "system", "content": "You infer relations between labeled components from label names only. Output strict JSON."},
        {"role": "user", "content": prompt_user},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = messages[0]["content"] + "\n\n" + messages[1]["content"] + "\n\nJSON:\n"

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
        )

    # Decode only generated continuation (avoid echoing prompt)
    input_len = inputs["input_ids"].shape[-1]
    gen_tokens = out[0][input_len:]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    return _robust_parse_json(decoded)


def _dedupe_and_validate(rel: Dict[str, Any], labels: List[str]) -> Dict[str, Any]:
    label_set = set(labels)

    def clean_pairs(pairs: Any) -> List[Dict[str, Any]]:
        if not isinstance(pairs, list):
            return []
        seen = set()
        out: List[Dict[str, Any]] = []
        for p in pairs:
            if not isinstance(p, dict):
                continue
            a = p.get("a")
            b = p.get("b")
            if not isinstance(a, str) or not isinstance(b, str):
                continue
            if a not in label_set or b not in label_set:
                continue
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)

            try:
                conf = float(p.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            evidence = p.get("evidence", "")
            if not isinstance(evidence, str):
                evidence = str(evidence)

            out.append({"a": a, "b": b, "confidence": conf, "evidence": evidence[:200]})
        return out

    neighboring = clean_pairs(rel.get("neighboring_pairs", []))
    related = clean_pairs(rel.get("same_pairs", []))  # "same_pairs" == related/similar

    notes = rel.get("notes", "")
    if not isinstance(notes, str):
        notes = str(notes)

    try:
        global_conf = float(rel.get("confidence", 0.0))
    except Exception:
        global_conf = 0.0
    global_conf = max(0.0, min(1.0, global_conf))

    return {
        "neighboring_pairs": neighboring,
        "same_pairs": related,
        "notes": notes[:400],
        "confidence": global_conf,
    }


def main():
    if not os.path.exists(REGISTRY_JSON):
        raise FileNotFoundError(f"Missing registry.json: {REGISTRY_JSON}")

    assembly_name = _read_assembly_name()

    print(f"[LOAD] registry: {REGISTRY_JSON}")
    if assembly_name != "Unknown":
        print(f"[LOAD] assembly_name: {assembly_name}")
    else:
        print("[LOAD] assembly_name: Unknown (components_inventory.json missing or no assembly_name)")

    registry = _load_json(REGISTRY_JSON)
    labels = _extract_labels_from_registry(registry)

    if len(labels) < 2:
        out_obj = {
            "model": MODEL_PATH,
            "assembly_name": assembly_name,
            "labels": labels,
            "neighboring_pairs": [],
            "same_pairs": [],
            "notes": "Not enough labels to infer relations.",
            "confidence": 0.0,
        }
        _save_json(OUT_RELATIONS_JSON, out_obj)
        print(f"[SAVE] {OUT_RELATIONS_JSON}")
        return

    print(f"[LLM] Loading {MODEL_PATH} ...")
    model, tokenizer = _load_llm(MODEL_PATH)

    print("[LLM] Inferring relations from labels + assembly_name ...")
    raw_rel = _call_llm_for_relations(model, tokenizer, assembly_name, labels)

    rel = _dedupe_and_validate(raw_rel, labels)

    out_obj = {
        "model": MODEL_PATH,
        "assembly_name": assembly_name,
        "labels": labels,
        **rel,
    }

    _save_json(OUT_RELATIONS_JSON, out_obj)
    print(f"[SAVE] {OUT_RELATIONS_JSON}")


if __name__ == "__main__":
    main()
