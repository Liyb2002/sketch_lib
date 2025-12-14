import os
import torch
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

def run_inference(base_path="."):
    print("\n" + "="*40)
    print("STEP 1: Inferring Component Inventory")
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

    print(f"Loading {model_path} on {device}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # --- LOAD IMAGES ---
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Master sketch not found at {master_path}")

    view_paths = sorted([os.path.join(views_path, f) for f in os.listdir(views_path) if f.endswith(('.png', '.jpg'))])[:6]
    images = [Image.open(master_path)] + [Image.open(p) for p in view_paths]

    # --- PROMPT ---
    system_prompt = """
    You are an Expert Structural Analyst. Analyze the Master Sketch and Views to break down the object into its constituent components.

    GUIDELINES:
    1. **Open Vocabulary:** Do NOT stick to a fixed list. Use the most precise name for whatever you see.
    2. **Be Exhaustive:** Identify ALL distinct parts. Do not generalize.
    3. **Universal Application:** The object can be anything (a vehicle, a gadget, a piece of furniture, a toy).

    Output strictly in JSON format:
    {
      "object_category": "Specific category",
      "components": [
        {"name": "Precise Part Name", "count": INTEGER_VALUE}
      ]
    }
    """

    # --- INFERENCE ---
    content_list = [{"type": "image", "image": img} for img in images]
    content_list.append({"type": "text", "text": system_prompt})
    
    messages = [{"role": "user", "content": content_list}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(device)

    print("Running inference...")
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    # --- SAVE ---
    clean_json = output_text.replace("```json", "").replace("```", "").strip()
    if "{" in clean_json:
        start = clean_json.find("{")
        end = clean_json.rfind("}") + 1
        clean_json = clean_json[start:end]

    try:
        parsed = json.loads(clean_json)
        with open(output_file, "w") as f:
            json.dump(parsed, f, indent=2)
        print(f"Success! Inventory saved to: {output_file}")
    except json.JSONDecodeError:
        print("Warning: JSON parsing failed. Saving raw text.")
        with open(output_file.replace(".json", ".txt"), "w") as f:
            f.write(clean_json)