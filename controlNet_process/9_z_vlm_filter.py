#!/usr/bin/env python3

import os
import json
import re
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"


# -----------------------------
# Utilities
# -----------------------------

def strip_name(name: str) -> str:
    return name.split("_")[0]


def normalize_pair(a, b):
    return tuple(sorted([a, b]))


def parse_pairs_from_text(text: str):
    """
    Only accept strict lines:
        partA - partB
    """
    pairs = set()
    for line in text.splitlines():
        line = line.strip().lower()
        m = re.match(r"^([a-zA-Z]+)\s*-\s*([a-zA-Z]+)$", line)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        pairs.add(normalize_pair(a, b))
    return pairs


# -----------------------------
# JSON loading
# -----------------------------

def load_json_relations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    relations = set()
    unknown_relations = set()

    for r in data.get("attachments", []):
        a = strip_name(r["a"])
        b = strip_name(r["b"])
        pair = normalize_pair(a, b)
        if a == "unknown" or b == "unknown":
            unknown_relations.add(pair)
        else:
            relations.add(pair)

    for r in data.get("containment", []):
        if "a" in r and "b" in r:
            a = strip_name(r["a"])
            b = strip_name(r["b"])
            pair = normalize_pair(a, b)
            if a == "unknown" or b == "unknown":
                unknown_relations.add(pair)
            else:
                relations.add(pair)

    return relations, unknown_relations


def extract_part_names(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    names = set()
    for n in data.get("nodes", {}).keys():
        s = strip_name(n)
        if s != "unknown":
            names.add(s)

    return sorted(names)


# -----------------------------
# VLM inference
# -----------------------------

def query_vlm_impossible_pairs(image_path, part_names):

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    parts_str = ", ".join(part_names)

    prompt = f"""
You are given an image of an object with multiple components.

Component names:
{parts_str}

Goal:
Output ONLY component pairs that are IMPOSSIBLE to be physically connected/touching/neighboring.

VERY IMPORTANT (be ultra conservative):
- Only include a pair if you are 100% certain it is impossible.
- If there is any doubt, any plausible hidden connection, or you are not fully sure: DO NOT include it.
- Prefer outputting too few pairs rather than any wrong pair.
- "Impossible" means: in a real object, these two parts could not be adjacent or touching given the structure.

Self-check:
Before outputting each pair, ask yourself: "Could these two parts possibly touch or connect in any plausible configuration of this object?"
If yes or maybe -> do NOT output it.
Only output if the answer is a definite NO.

Rules:
- Output ONLY the pairs (no explanation).
- One pair per line.
- Format exactly: partA - partB
- No bullets, no numbering, no extra text.
"""

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=chat_text,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    text = processor.decode(outputs[0], skip_special_tokens=True)
    return text


# -----------------------------
# Main
# -----------------------------

def main():
    root = os.path.dirname(os.path.abspath(__file__))

    json_path = os.path.join(root, "sketch", "AEP", "initial_constraints.json")
    image_path = os.path.join(root, "sketch", "input.png")

    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    # Load JSON relations (excluding unknown)
    json_relations, unknown_relations = load_json_relations(json_path)
    part_names = extract_part_names(json_path)

    print("\n==============================")
    print("STEP 1: JSON relations (excluding unknown)")
    print("==============================")
    for a, b in sorted(json_relations):
        print(f"{a} - {b}")

    print("\n==============================")
    print("UNKNOWN relations (ignored by VLM, kept by default)")
    print("==============================")
    for a, b in sorted(unknown_relations):
        print(f"{a} - {b}")

    # Query VLM for ultra-sure impossible relations
    print("\n==============================")
    print("STEP 2: Querying VLM for ULTRA-SURE impossible relations...")
    print("==============================")

    vlm_raw = query_vlm_impossible_pairs(image_path, part_names)

    print("\nRaw VLM output:\n")
    print(vlm_raw)

    vlm_impossible = parse_pairs_from_text(vlm_raw)

    # Restrict to known parts only (extra safety)
    valid_parts = set(part_names)
    vlm_impossible = {
        (a, b) for (a, b) in vlm_impossible
        if a in valid_parts and b in valid_parts
    }

    print("\n==============================")
    print("STEP 3: Parsed VLM impossible relations (ultra-sure)")
    print("==============================")
    if not vlm_impossible:
        print("None")
    else:
        for a, b in sorted(vlm_impossible):
            print(f"{a} - {b}")

    # Reverse filtering: remove only those that VLM says impossible
    to_remove = json_relations & vlm_impossible
    safe_relations = json_relations - to_remove

    print("\n==============================")
    print("STEP 4: Relations to REMOVE (JSON ∩ VLM-impossible)")
    print("==============================")
    if not to_remove:
        print("None")
    else:
        for a, b in sorted(to_remove):
            print(f"{a} - {b}")

    print("\n==============================")
    print("STEP 5: Relations KEPT")
    print("==============================")
    for a, b in sorted(safe_relations):
        print(f"{a} - {b}")

    print("\nDone.")


if __name__ == "__main__":
    main()
