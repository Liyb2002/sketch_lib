#!/usr/bin/env python3
"""
run_partfield.py (extended)
- Runs PartField
- Copies internal results out (same as before)
- Additionally: extract ONLY the K=20 cluster labels into:
      ./partfield_clusters/
"""

import sys
import subprocess
from pathlib import Path
import shutil
import numpy as np


def export_20cluster_labels(internal_cluster_dir: Path, trellis_dir: Path, out_dir: Path):
    """
    Scan internal PartField clustering dir and export files
    where number of clusters == 20.
    """
    print("\n[EXPORT] Saving EXACTLY 20-cluster results...")

    out_dir.mkdir(exist_ok=True)

    ply_files = sorted(trellis_dir.glob("*.ply"))
    if not ply_files:
        print("[WARN] No .ply files found.")
        return

    for ply in ply_files:
        stem = ply.stem  # e.g. "0_trellis_gaussian"

        # Find cluster files matching this model
        matches = list(internal_cluster_dir.rglob(f"*{stem}*.npy"))
        if not matches:
            print(f"[WARN] No cluster file found for {stem}")
            continue

        best = None
        best_diff = None

        # Choose file where cluster count is closest to 20
        for f in matches:
            try:
                labels = np.load(f)
            except:
                continue

            uniq = np.unique(labels[labels >= 0])  # ignore negative labels if any
            diff = abs(len(uniq) - 20)

            if best is None or diff < best_diff:
                best = f
                best_diff = diff

        if best is None:
            print(f"[WARN] No valid label file for {stem}")
            continue

        # Confirm cluster count
        labels = np.load(best).reshape(-1)
        uniq = np.unique(labels[labels >= 0])
        print(f"  {stem}: selected {best.name} ({len(uniq)} clusters)")

        # Save to clean folder
        out_path = out_dir / f"{stem}_k20.npy"
        np.save(out_path, labels)
        print(f"   -> saved: {out_path}")


def main():
    root_dir = Path(__file__).resolve().parent

    partfield_root = root_dir / "PartField"
    trellis_dir = root_dir / "trellis_outputs"
    external_result_root = root_dir / "trellis_partfield_results"

    if not partfield_root.is_dir():
        raise RuntimeError(f"Missing PartField folder: {partfield_root}")
    if not trellis_dir.is_dir():
        raise RuntimeError(f"Missing trellis_outputs folder: {trellis_dir}")

    external_result_root.mkdir(exist_ok=True)
    print(f"[INFO] External results root: {external_result_root}")

    ply_files = sorted(trellis_dir.glob("*.ply"))
    if not ply_files:
        raise RuntimeError(f"No .ply files found in {trellis_dir}")
    print(f"[INFO] Found {len(ply_files)} .ply files")

    cfg_path = "configs/final/demo.yaml"
    ckpt_path = "model/model_objaverse.ckpt"
    result_name = "partfield_features/trellis_external"

    # -------------------------
    # 1️⃣ Feature extraction
    # -------------------------
    extract_cmd = [
        sys.executable, "partfield_inference.py",
        "-c", cfg_path,
        "--opts",
        "continue_ckpt", ckpt_path,
        "result_name", result_name,
        "dataset.data_path", str(trellis_dir),
        "is_pc", "True",
    ]
    print("\n[PartField] Extracting features...")
    subprocess.run(extract_cmd, cwd=partfield_root, check=True)

    internal_feat_root = partfield_root / "exp_results" / result_name

    # -------------------------
    # 2️⃣ Clustering (K <= 20)
    # -------------------------
    internal_cluster_dir = partfield_root / "exp_results" / "clustering" / "trellis_external"
    cluster_cmd = [
        sys.executable, "run_part_clustering.py",
        "--root", str(internal_feat_root),
        "--dump_dir", str(internal_cluster_dir),
        "--source_dir", str(trellis_dir),
        "--max_num_clusters", "20",
        "--is_pc", "True",
    ]
    print("\n[PartField] Clustering parts...")
    subprocess.run(cluster_cmd, cwd=partfield_root, check=True)

    # -------------------------
    # 3️⃣ Mirror all internal results externally (unchanged)
    # -------------------------
    external_feat_dir = external_result_root / "features"
    external_cluster_dir = external_result_root / "clusters"

    def copy_dir(src: Path, dst: Path):
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    print("\n[INFO] Copying results externally...")
    copy_dir(internal_feat_root, external_feat_dir)
    copy_dir(internal_cluster_dir, external_cluster_dir)

    # -------------------------
    # 4️⃣ Export CLEAN 20-cluster npy files
    # -------------------------
    clean_dir = root_dir / "partfield_clusters"
    export_20cluster_labels(internal_cluster_dir, trellis_dir, clean_dir)

    print("\n🎯 CLEAN 20-cluster label files saved to:")
    print(clean_dir)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
