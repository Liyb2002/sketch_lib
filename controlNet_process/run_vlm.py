import os
import sys

# Add the current directory to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the modules from the new folder
from vlm_pipeline import infer_components, write_dsl

def main():
    base_path = os.getcwd()
    
    # 1. Ensure output directory exists
    program_dir = os.path.join(base_path, "sketch", "program")
    os.makedirs(program_dir, exist_ok=True)
    
    print(f"Project Base Path: {base_path}")
    print(f"Output Directory: {program_dir}")
    
    try:
        # 2. Run Step 1: Inventory
        infer_components.run_inference(base_path)
        
        # 3. Run Step 2: DSL Draft
        write_dsl.run_dsl_generation(base_path)
        
        print("\n" + "="*40)
        print("PIPELINE COMPLETE")
        print("Check 'sketch/program/' for the generated files.")
        print("="*40)
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")

if __name__ == "__main__":
    main()