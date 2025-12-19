import os
import torch
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

def run_inference(base_path="."):
    print("\n" + "="*40)
    print("STEP 1: Structural Decomposition (Non-Overlapping)")
    print("="*40)

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
    view_paths = sorted([os.path.join(views_path, f) for f in os.listdir(views_path) if f.endswith(('.png', '.jpg'))])[:6]
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
    generated_ids = model.generate(**inputs, max_new_tokens=1536) # Increased tokens for more detail
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    # --- CLEAN & SAVE ---
    clean_json = output_text.replace("```json", "").replace("```", "").strip()
    if "{" in clean_json:
        start = clean_json.find("{")
        end = clean_json.rfind("}") + 1
        clean_json = clean_json[start:end]

    try:
        parsed = json.loads(clean_json)
        with open(output_file, "w") as f:
            json.dump(parsed, f, indent=2)
        print(f"Success! Decomposition saved to: {output_file}")
    except json.JSONDecodeError:
        with open(output_file.replace(".json", ".txt"), "w") as f:
            f.write(clean_json)