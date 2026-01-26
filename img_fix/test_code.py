import os
from google import genai
from google.genai.types import (
    EditImageConfig,
    Image,
    MaskReferenceConfig,
    MaskReferenceImage,
    RawReferenceImage,
)

# 1. Setup Client
PROJECT_ID = "880503416167" # Replace with your actual project ID
LOCATION = "us-central1"
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

def merge_sketches_unified(input_path, mask_path, output_name):
    print(f"Processing: {input_path}")
    
    # Use the factory method from your provided code to load images
    initial_image = Image.from_file(location=input_path)
    initial_mask = Image.from_file(location=mask_path)

    # 2. Define the Reference Images (The New Standard)
    # reference_id 0 is the base image, 1 is the mask
    raw_ref_image = RawReferenceImage(
        reference_image=initial_image, 
        reference_id=0
    )
    
    mask_ref_image = MaskReferenceImage(
        reference_id=1,
        reference_image=initial_mask,
        config=MaskReferenceConfig(
            mask_mode="MASK_MODE_USER_PROVIDED",
            mask_dilation=0.03, # Critical for bridging sketch lines
        ),
    )

    # 3. Call the Edit Model
    # Note: Inpainting uses "EDIT_MODE_INPAINT_INSERTION"
    edit_prompt = (
        "A seamless pencil sketch transition. Connect the lines perfectly "
        "using black and white pencil strokes, maintain hand-drawn style."
    )
    
    response = client.models.edit_image(
        model="imagen-3.0-capability-001",
        prompt=edit_prompt,
        reference_images=[raw_ref_image, mask_ref_image],
        config=EditImageConfig(
            edit_mode="EDIT_MODE_INPAINT_INSERTION",
            number_of_images=1,
            negative_prompt="color, realistic, photo, 3d, shading, blurry",
            safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
        ),
    )

    # 4. Save result
    # Accessing ._pil_image as seen in your display_images sample
    res_img = response.generated_images[0].image._pil_image
    res_img.save(f"{output_name}.png")
    print(f"Saved merged sketch to {output_name}.png")

if __name__ == "__main__":
    merge_sketches_unified(
        input_path='input.png', 
        mask_path='mask.png', 
        output_name='final_sketch_bridge'
    )