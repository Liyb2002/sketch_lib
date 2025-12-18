import os
import torch
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

def run_dsl_generation(base_path="."):
    print("\n" + "="*40)
    print("STEP 2: Spatial & Symmetry Reasoning")
    print("="*40)

    # --- PATH SETUP ---
    program_dir = os.path.join(base_path, "sketch", "program")
    # Using the decomposition file from the previous step
    inventory_path = os.path.join(program_dir, "components_inventory.json")
    master_sketch_path = os.path.join(base_path, "sketch", "input.png")
    output_file = os.path.join(program_dir, "dsl_draft.json")

    # --- LOAD MODEL ---
    model_path = "Qwen/Qwen2-VL-7B-Instruct" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {model_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # --- LOAD DATA ---
    if not os.path.exists(inventory_path):
        raise FileNotFoundError(f"components_inventory JSON not found at {inventory_path}.")
    if not os.path.exists(master_sketch_path):
        raise FileNotFoundError(f"Master sketch not found at {master_sketch_path}.")

    with open(inventory_path, "r") as f:
        inventory_data = json.load(f)
        inventory_context = json.dumps(inventory_data, indent=2)

    # Use the Master Sketch as the primary visual for spatial reasoning
    master_img = Image.open(master_sketch_path)

    # --- PROMPT: REASONING TASKS ---
    dsl_prompt = f"""
    You are a Geometric Logic Engine. Your task is to analyze the Master Sketch (input.png) and the provided JSON Inventory to build a relational model of the assembly.

    INPUT CONTEXT (Component Inventory):
    {inventory_context}

    TASK:
    1. **Neighboring Relations (Connectivity):** Identify which parts are physically connected or touching in the sketch.
    2. **Equivalence Relations (Type Identity):** Identify distinct part IDs that are the same component type (e.g., if there are 2 wheels, 'wheel_0' and 'wheel_1' are Equivalent).

    CRITICAL RULES:
    - **Strict Vocabulary:** Use ONLY names from the INPUT CONTEXT.
    - **Unique Instantiation:** Generate unique IDs for every individual part based on the "count" (e.g., 'Leg' count 2 becomes 'leg_0' and 'leg_1').
    - **Logical Flow:** Reason Neighboring FIRST, then Equivalence.

    OUTPUT FORMAT (JSON ONLY):
    {{
      "reasoning": {{
        "neighbor_logic": "Explain which parts are touching based on the sketch.",
        "equivalence_logic": "Explain which IDs are instances of the same part type."
      }},
      "assembly": {{
        "instances": [
          {{ "id": "part_id_0", "type": "original_name_from_json" }}
        ],
        "connectivity_graph": [
          {{ "source": "part_id_0", "target": "part_id_1", "connection_type": "adjacent" }}
        ],
        "equivalence_groups": [
          {{ "type_name": "original_name", "member_ids": ["part_id_0", "part_id_1"] }}
        ]
      }}
    }}
    """

    # --- INFERENCE ---
    # We focus specifically on the master sketch for spatial layout
    content_list = [
        {"type": "image", "image": master_img},
        {"type": "text", "text": dsl_prompt}
    ]

    messages = [{"role": "user", "content": content_list}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[master_img], padding=True, return_tensors="pt").to(device)

    print("Reasoning spatial relations...")
    generated_ids = model.generate(**inputs, max_new_tokens=2000)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_dsl = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    # --- CLEAN & SAVE ---
    clean_dsl = output_dsl.replace("```json", "").replace("```", "").strip()
    if "{" in clean_dsl:
        start = clean_dsl.find("{")
        end = clean_dsl.rfind("}") + 1
        clean_dsl = clean_dsl[start:end]

    try:
        parsed = json.loads(clean_dsl)
        with open(output_file, "w") as f:
            json.dump(parsed, f, indent=2)
        print(f"Success! Relational Graph saved to: {output_file}")
    except json.JSONDecodeError:
        print("JSON Error. Saving raw text.")
        with open(output_file.replace(".json", ".txt"), "w") as f:
            f.write(clean_dsl)