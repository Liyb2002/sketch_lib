#!/usr/bin/env python3
"""
run_partfield.py (Customized)
- Inputs: Single mesh at sketch/3d_reconstruction/fused_model.ply
- Action: Runs PartField feature extraction and clustering
- Output: Saves ONLY the k=20 labels to sketch/3d_reconstruction/clusters_k20.npy
"""

import sys
import subprocess
from pathlib import Path
import shutil
import numpy as np

# --- CONFIG ---
ROOT_DIR = Path(__file__).resolve().parent
PARTFIELD_ROOT = ROOT_DIR / "PartField"  # Assumes PartField repo is here
TARGET_PLY = ROOT_DIR / "sketch" / "3d_reconstruction" / "fused_model.ply"
TARGET_DIR = TARGET_PLY.parent # sketch/3d_reconstruction

def export_20cluster_labels(internal_cluster_dir: Path, out_dir: Path):
    """
    Find the cluster file closest to 20 clusters and save it.
    """
    print("\n[EXPORT] Saving 20-cluster results...")

    # PartField names files based on the input ply stem
    stem = TARGET_PLY.stem  # "fused_model"

    # Find matches inside PartField's internal output folder
    # Usually named like: fused_model_ins_seg_hier_spectral_19.npy
    matches = list(internal_cluster_dir.rglob(f"*{stem}*.npy"))
    
    if not matches:
        print(f"[ERROR] No cluster files found for {stem} in {internal_cluster_dir}")
        return

    best_file = None
    best_diff = float('inf')
    best_k = 0

    # Logic: Find file with cluster count closest to 20
    for f in matches:
        try:
            labels = np.load(f)
            # Filter valid labels (>=0)
            unique_clusters = len(np.unique(labels[labels >= 0]))
            
            diff = abs(unique_clusters - 20)
            
            # Prefer exactly 20, or the closest one
            if diff < best_diff:
                best_diff = diff
                best_file = f
                best_k = unique_clusters
            elif diff == best_diff and unique_clusters > best_k:
                # Tie-breaker: prefer more clusters if equidistant (e.g. 18 vs 22, pick 22)
                best_file = f
                best_k = unique_clusters
                
        except Exception as e:
            print(f"[WARN] Failed to load {f}: {e}")
            continue

    if best_file:
        print(f"  Selected: {best_file.name} (K={best_k})")
        
        # Save to the target directory
        out_path = out_dir / "clusters_k20.npy"
        
        labels = np.load(best_file).reshape(-1)
        np.save(out_path, labels)
        print(f"   -> Saved to: {out_path}")
    else:
        print("[ERROR] Could not determine a valid cluster file.")


def main():
    if not PARTFIELD_ROOT.is_dir():
        raise RuntimeError(f"PartField repo not found at: {PARTFIELD_ROOT}")
    
    if not TARGET_PLY.exists():
        raise RuntimeError(f"Input PLY not found at: {TARGET_PLY}")

    print(f"[INFO] Processing: {TARGET_PLY}")

    # Internal PartField config paths
    cfg_path = "configs/final/demo.yaml"
    ckpt_path = "model/model_objaverse.ckpt"
    result_name = "partfield_features/single_inference"

    # -------------------------
    # 1️⃣ Feature Extraction
    # -------------------------
    # Note: PartField expects a folder, so we point to the parent folder
    # It might try to process all .plys in there, but we only care about fused_model
    extract_cmd = [
        sys.executable, "partfield_inference.py",
        "-c", cfg_path,
        "--opts",
        "continue_ckpt", ckpt_path,
        "result_name", result_name,
        "dataset.data_path", str(TARGET_DIR), # Points to sketch/3d_reconstruction
        "is_pc", "True",
    ]
    
    print("\n[PartField] Extracting features...")
    # Running inside PartField dir so imports work
    subprocess.run(extract_cmd, cwd=PARTFIELD_ROOT, check=True)

    internal_feat_root = PARTFIELD_ROOT / "exp_results" / result_name

    # -------------------------
    # 2️⃣ Clustering (Target K=20)
    # -------------------------
    internal_cluster_dir = PARTFIELD_ROOT / "exp_results" / "clustering" / "single_inference"
    
    cluster_cmd = [
        sys.executable, "run_part_clustering.py",
        "--root", str(internal_feat_root),
        "--dump_dir", str(internal_cluster_dir),
        "--source_dir", str(TARGET_DIR),
        "--max_num_clusters", "20", # Force optimization towards 20
        "--is_pc", "True",
    ]
    
    print("\n[PartField] Clustering parts...")
    subprocess.run(cluster_cmd, cwd=PARTFIELD_ROOT, check=True)

    # -------------------------
    # 3️⃣ Export ONLY the K=20 file
    # -------------------------
    export_20cluster_labels(internal_cluster_dir, TARGET_DIR)

    print("\n✅ Done.")

if __name__ == "__main__":
    main()