import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from constraints_extraction import pca_analysis, vis_pca_bbx

# --- PATH CONFIG ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.join(ROOT_DIR, "sketch", "3d_reconstruction", "final_segmentation")

# Inputs
INPUT_PLY = os.path.join(SCENE_DIR, "semantic_fused_model.ply")
INPUT_REGISTRY = os.path.join(SCENE_DIR, "segmentation_registry.json")

# Outputs
OUTPUT_DIR = os.path.join(ROOT_DIR, "sketch", "program", "dsl_extraction")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("="*60)
    print("STARTING PARSEL EXTRACTION PIPELINE")
    print("="*60)

    # -------------------------------------
    # STEP 1: PCA Analysis (Box Fitting)
    # -------------------------------------
    try:
        parts_db = pca_analysis.run_pca_analysis(
            ply_path=INPUT_PLY, 
            registry_path=INPUT_REGISTRY
        )
        
        # Save output
        step1_json = os.path.join(OUTPUT_DIR, "step1_pca_primitives.json")
        pca_analysis.save_primitives_to_json(parts_db, step1_json)
        
    except Exception as e:
        print(f"[CRITICAL ERROR] PCA Analysis failed: {e}")
        return

    # -------------------------------------
    # DEBUG: Visualization
    # -------------------------------------
    # Comment this out later if running in batch mode
    vis_pca_bbx.run_visualization(step1_json)

    print("="*60)
    print("Done.")

if __name__ == "__main__":
    main()