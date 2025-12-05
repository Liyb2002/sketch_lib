#!/usr/bin/env python3
"""
run_partfield.py

Run PartField on .ply files in ./trellis_outputs (point cloud mode),
then copy the results to ./trellis_partfield_results/ at the same level.
"""

import sys
import subprocess
from pathlib import Path
import shutil


def main():
    # This script sits at same level as PartField/ and trellis_outputs/
    root_dir = Path(__file__).resolve().parent

    partfield_root = root_dir / "PartField"
    trellis_dir = root_dir / "trellis_outputs"
    external_result_root = root_dir / "trellis_partfield_results"

    if not partfield_root.is_dir():
        raise RuntimeError(f"Cannot find PartField folder: {partfield_root}")
    if not trellis_dir.is_dir():
        raise RuntimeError(f"Cannot find trellis_outputs folder: {trellis_dir}")

    # Make external results folder (where YOU want things)
    external_result_root.mkdir(exist_ok=True)
    print(f"[INFO] External results root: {external_result_root}")

    # Sanity check: at least one .ply
    ply_files = sorted(trellis_dir.glob("*.ply"))
    if not ply_files:
        raise RuntimeError(f"No .ply files found in {trellis_dir}")
    print(f"[INFO] Found {len(ply_files)} .ply files")

    # Paths relative to PartField root
    cfg_path = "configs/final/demo.yaml"
    ckpt_path = "model/model_objaverse.ckpt"

    # This must stay RELATIVE, PartField will put it under exp_results/
    result_name = "partfield_features/trellis_external"

    # -------------------------
    # 1️⃣ Feature extraction
    # -------------------------
    extract_cmd = [
        sys.executable,
        "partfield_inference.py",
        "-c", cfg_path,
        "--opts",
        "continue_ckpt", ckpt_path,
        "result_name", result_name,           # relative, used under exp_results/
        "dataset.data_path", str(trellis_dir),
        "is_pc", "True",
    ]

    print("\n[PartField] Extracting features...")
    print("  CWD:", partfield_root)
    print("  CMD:", " ".join(extract_cmd))
    subprocess.run(extract_cmd, cwd=partfield_root, check=True)

    # Internal PartField feature dir (where it actually wrote)
    internal_feat_root = partfield_root / "exp_results" / result_name

    # -------------------------
    # 2️⃣ Clustering
    # -------------------------
    internal_cluster_dir_rel = "exp_results/clustering/trellis_external"
    internal_cluster_dir = partfield_root / internal_cluster_dir_rel

    cluster_cmd = [
        sys.executable,
        "run_part_clustering.py",
        "--root", str(internal_feat_root),
        "--dump_dir", str(internal_cluster_dir),
        "--source_dir", str(trellis_dir),
        "--max_num_clusters", "20",
        "--is_pc", "True",
    ]

    print("\n[PartField] Clustering parts...")
    print("  CWD:", partfield_root)
    print("  CMD:", " ".join(cluster_cmd))
    subprocess.run(cluster_cmd, cwd=partfield_root, check=True)

    # -------------------------
    # 3️⃣ Mirror results outside PartField
    # -------------------------
    external_feat_dir = external_result_root / "features"
    external_cluster_dir = external_result_root / "clusters"

    # copytree with overwrite support
    def copy_dir(src: Path, dst: Path):
        if dst.exists():
            # remove old to avoid nested copies
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    print("\n[INFO] Copying results externally...")
    print(f"  Features:  {internal_feat_root} -> {external_feat_dir}")
    print(f"  Clusters:  {internal_cluster_dir} -> {external_cluster_dir}")

    copy_dir(internal_feat_root, external_feat_dir)
    copy_dir(internal_cluster_dir, external_cluster_dir)

    print("\n✅ Done.")
    print(f"External features dir:  {external_feat_dir}")
    print(f"External clusters dir:  {external_cluster_dir}")


if __name__ == "__main__":
    main()
