import os
import torch
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

def run_dsl_generation(base_path="."):
    print("\n" + "="*40)
    print("STEP 2: Generating ParSEL DSL Draft")
    print("="*40)

    # --- PATH SETUP ---
    program_dir = os.path.join(base_path, "sketch", "program")
    inventory_path = os.path.join(program_dir, "components_inventory.json")
    views_path = os.path.join(base_path, "sketch", "views")
    output_file = os.path.join(program_dir, "dsl_draft.json")

    # --- LOAD MODEL ---
    # Note: In a production script, you'd pass the model object to avoid reloading 
    # but for simplicity, we load again or rely on OS caching.
    model_path = "Qwen/Qwen2-VL-7B-Instruct" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if we can reuse a global model if you merge this logic, 
    # but strictly following the request for modular files:
    print(f"Loading {model_path} on {device}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # --- LOAD DATA ---
    if not os.path.exists(inventory_path):
        raise FileNotFoundError(f"Inventory not found at {inventory_path}. Run Step 1 first.")

    with open(inventory_path, "r") as f:
        inventory_context = f.read()

    image_paths = sorted([os.path.join(views_path, f) for f in os.listdir(views_path) if f.endswith(('.png', '.jpg'))])[:6]
    images = [Image.open(p) for p in image_paths]

    # --- PROMPT ---
    dsl_prompt = f"""
    You are a Geometric Architect. 
    INPUT CONTEXT: The user has verified the following component inventory:
    {inventory_context}

    YOUR TASK:
    Convert this inventory into a hierarchical ParSEL DSL Graph.

    CRITICAL RULES:
    1. **Strict Vocabulary:** You must ONLY use the component names listed in the INPUT CONTEXT.
    2. **Instantiation:** If the inventory says "count: 4", you must generate 4 distinct IDs (e.g., "component_0", "component_1").
    4. **Symmetry:** Group identical parts into "symmetry_groups".
    5. **Connectivity:** Infer logical attachments using abstract faces: top, bottom, left, right, front, back.

    OUTPUT FORMAT (JSON ONLY):
    {{
      "dsl_version": "1.0",
      "root_part": "ROOT_ID",
      "parts": [
        {{ "id": "name_0", "semantic": "name_from_inventory", "primitive": "cuboid" }}
      ],
      "symmetry_groups": [
        {{ "name": "group_name", "members": ["name_0", "name_1"], "type": "reflective" }}
      ],
      "constraints": [
        {{ "type": "attach", "source": "name_1", "source_face": "top", "target": "name_0", "target_face": "bottom" }}
      ]
    }}
    """

    # --- INFERENCE ---
    content_list = [{"type": "image", "image": img} for img in images]
    content_list.append({"type": "text", "text": dsl_prompt})

    messages = [{"role": "user", "content": content_list}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(device)

    print("Generating DSL...")
    generated_ids = model.generate(**inputs, max_new_tokens=1500)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_dsl = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    # --- SAVE ---
    clean_dsl = output_dsl.replace("```json", "").replace("```", "").strip()
    if "{" in clean_dsl:
        start = clean_dsl.find("{")
        end = clean_dsl.rfind("}") + 1
        clean_dsl = clean_dsl[start:end]

    try:
        parsed = json.loads(clean_dsl)
        with open(output_file, "w") as f:
            json.dump(parsed, f, indent=2)
        print(f"Success! DSL Draft saved to: {output_file}")
    except json.JSONDecodeError:
        print("Warning: Output is not valid JSON. Saving raw text.")
        with open(output_file.replace(".json", ".txt"), "w") as f:
            f.write(clean_dsl)