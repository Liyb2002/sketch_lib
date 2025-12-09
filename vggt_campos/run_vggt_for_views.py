import os
import sys
import torch
import numpy as np
import cv2
import json

# -----------------------------------------------------------------------------
# 1. SETUP PATHS & IMPORTS
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VGGT_REPO_ROOT = os.path.join(THIS_DIR, "vggt")
sys.path.insert(0, VGGT_REPO_ROOT)

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except ImportError as e:
    print(f"[!] Error importing VGGT modules: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 2. CONFIGURATION
# -----------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(THIS_DIR, "vggt_checkpoint.pth")
VIEWS_DIR = os.path.join(THIS_DIR, "views")
OUTPUT_DIR = os.path.join(THIS_DIR, "outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHITE_THRESHOLD = 250 

def save_camera_params(save_path, extrinsics, intrinsics):
    data = {
        "extrinsics": extrinsics.tolist(),
        "intrinsics": intrinsics.tolist()
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Running on device: {DEVICE}")

    # -------------------------------------------------------------------------
    # 3. LOAD MODEL
    # -------------------------------------------------------------------------
    print("Loading VGGT model structure...")
    model = VGGT()
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[!] Checkpoint not found at {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Handle checkpoint format
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: state_dict = checkpoint['model']
        else: state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean keys
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    # -------------------------------------------------------------------------
    # 4. LOAD IMAGES
    # -------------------------------------------------------------------------
    image_files = sorted([f for f in os.listdir(VIEWS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print(f"No images found in {VIEWS_DIR}")
        return

    print(f"Found {len(image_files)} images. Preprocessing...")
    image_paths = [os.path.join(VIEWS_DIR, f) for f in image_files]
    
    # Returns [Views, 3, H, W]
    input_tensor = load_and_preprocess_images(image_paths)
    # Add batch dim -> [1, Views, 3, H, W]
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    # -------------------------------------------------------------------------
    # 5. INFERENCE
    # -------------------------------------------------------------------------
    print("Running inference...")
    with torch.no_grad():
        preds = model(input_tensor)

    pred_depths = preds['depth']       # [B, V, H, W]
    pred_pose_enc = preds['pose_enc']  # [B, V, Enc_Dim]

    # FIX: Get H, W and pass them to the converter
    H, W = input_tensor.shape[-2], input_tensor.shape[-1]
    pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(pred_pose_enc, (H, W))

    # -------------------------------------------------------------------------
    # 6. SAVE OUTPUTS
    # -------------------------------------------------------------------------
    print("Processing and saving outputs...")
    num_views = input_tensor.shape[1]
    
    for v in range(num_views):
        view_name = os.path.splitext(image_files[v])[0]
        
        # --- Depth ---
        depth_map = pred_depths[0, v].cpu().numpy()
        orig_img = cv2.imread(image_paths[v])
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if depth_map.shape[:2] != orig_img.shape[:2]:
            depth_map = cv2.resize(depth_map, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # White Mask
        white_mask = np.all(orig_img >= WHITE_THRESHOLD, axis=-1)
        depth_map_masked = depth_map.copy()
        depth_map_masked[white_mask] = np.nan

        np.save(os.path.join(OUTPUT_DIR, f"{view_name}_depth.npy"), depth_map_masked)
        
        # Vis
        valid_depth = depth_map_masked[~np.isnan(depth_map_masked)]
        if valid_depth.size > 0:
            d_min, d_max = valid_depth.min(), valid_depth.max()
            depth_vis = (depth_map_masked - d_min) / (d_max - d_min + 1e-8)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            depth_vis[white_mask] = 0
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{view_name}_depth_vis.png"), depth_vis)

        # --- Camera ---
        extrinsic = pred_extrinsics[0, v].cpu().numpy()
        intrinsic = pred_intrinsics[0, v].cpu().numpy()
        
        save_camera_params(
            os.path.join(OUTPUT_DIR, f"{view_name}_cam.json"),
            extrinsic,
            intrinsic
        )

    print(f"Done! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()