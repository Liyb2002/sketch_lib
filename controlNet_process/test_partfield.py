#!/usr/bin/env python3
"""
run_partfield_single.py

All paths are RELATIVE to this script's folder (ROOT).

- PartField package: ROOT/packages/PartField
- Input point cloud: ROOT/sketch/3d_reconstruction/fused_model.ply
- Output folder:     ROOT/sketch/partfield_test/

Outputs:
  1) Clean K=20 labels:
     ROOT/sketch/partfield_test/fused_model_k20.npy

  2) Mirrored PartField internal outputs:
     ROOT/sketch/partfield_test/partfield_features/
     ROOT/sketch/partfield_test/partfield_clusters_raw/

  3) A local dataset folder used as PartField input:
     ROOT/sketch/partfield_test/partfield_input/fused_model.ply
"""

import sys
import subprocess
from pathlib import Path
import shutil
import numpy as np


def copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError(f"Missing source dir to copy: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def export_k20_for_single_model(internal_cluster_dir: Path, stem: str, out_npy_path: Path) -> None:
    """
    Find all .npy cluster label files matching `stem` and select the one whose
    number of clusters is closest to 20. Save to out_npy_path.
    """
    print("\n[EXPORT] Selecting best K≈20 cluster labels for:", stem)

    matches = list(internal_cluster_dir.rglob(f"*{stem}*.npy"))
    if not matches:
        raise RuntimeError(
            f"No cluster .npy files found matching stem '{stem}' under: {internal_cluster_dir}\n"
            f"Tip: check if clustering actually produced outputs, and inspect that folder."
        )

    best = None
    best_diff = None
    best_k = None

    for f in matches:
        try:
            labels = np.load(f).reshape(-1)
        except Exception:
            continue

        uniq = np.unique(labels[labels >= 0])  # ignore negative labels if any
        k = int(len(uniq))
        diff = abs(k - 20)

        if best is None or diff < best_diff:
            best = f
            best_diff = diff
            best_k = k

    if best is None:
        raise RuntimeError(
            f"Found cluster files for '{stem}', but none could be loaded as numpy arrays."
        )

    labels = np.load(best).reshape(-1)
    uniq = np.unique(labels[labels >= 0])
    print(f"[EXPORT] Selected: {best.name} ({len(uniq)} clusters)")

    out_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy_path, labels)
    print(f"[EXPORT] Saved clean labels to: {out_npy_path}")


def main():
    ROOT = Path(__file__).resolve().parent

    # -------------------------
    # Relative paths (as requested)
    # -------------------------
    PARTFIELD_ROOT = ROOT / "packages" / "PartField"
    IN_PLY = ROOT / "sketch" / "3d_reconstruction" / "fused_model.ply"
    OUT_ROOT = ROOT / "sketch" / "partfield_test"

    if not PARTFIELD_ROOT.is_dir():
        raise RuntimeError(f"Missing PartField folder: {PARTFIELD_ROOT}")
    if not IN_PLY.is_file():
        raise RuntimeError(f"Missing input ply: {IN_PLY}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] ROOT: {ROOT}")
    print(f"[INFO] PartField root: {PARTFIELD_ROOT}")
    print(f"[INFO] Input ply: {IN_PLY}")
    print(f"[INFO] Output root: {OUT_ROOT}")

    # PartField expects a directory for dataset.data_path, so create one and copy the ply in.
    input_dir = OUT_ROOT / "partfield_input"
    input_dir.mkdir(parents=True, exist_ok=True)

    input_ply = input_dir / IN_PLY.name
    # Copy only if not already exactly the same file
    if input_ply.resolve() != IN_PLY.resolve():
        shutil.copy2(IN_PLY, input_ply)

    stem = input_ply.stem  # fused_model
    print(f"[INFO] PartField dataset dir: {input_dir}")
    print(f"[INFO] Dataset file: {input_ply}")

    # -------------------------
    # PartField config (relative to PartField repo)
    # -------------------------
    cfg_path = "configs/final/demo.yaml"
    ckpt_path = "model/model_objaverse.ckpt"

    # Use unique result names to avoid collisions
    result_name = "partfield_features/partfield_test_single"
    clustering_name = "partfield_test_single"

    # -------------------------
    # 1) Feature extraction
    # -------------------------
    extract_cmd = [
        sys.executable, "partfield_inference.py",
        "-c", cfg_path,
        "--opts",
        "continue_ckpt", ckpt_path,
        "result_name", result_name,
        "dataset.data_path", str(input_dir),
        "is_pc", "True",
    ]
    print("\n[PartField] Extracting features...")
    subprocess.run(extract_cmd, cwd=str(PARTFIELD_ROOT), check=True)

    internal_feat_root = PARTFIELD_ROOT / "exp_results" / result_name
    if not internal_feat_root.exists():
        raise RuntimeError(f"Expected feature dir not found: {internal_feat_root}")

    # -------------------------
    # 2) Clustering (K <= 20)
    # -------------------------
    internal_cluster_dir = PARTFIELD_ROOT / "exp_results" / "clustering" / clustering_name
    cluster_cmd = [
        sys.executable, "run_part_clustering.py",
        "--root", str(internal_feat_root),
        "--dump_dir", str(internal_cluster_dir),
        "--source_dir", str(input_dir),
        "--max_num_clusters", "20",
        "--is_pc", "True",
    ]
    print("\n[PartField] Clustering parts...")
    subprocess.run(cluster_cmd, cwd=str(PARTFIELD_ROOT), check=True)

    if not internal_cluster_dir.exists():
        raise RuntimeError(f"Expected clustering dir not found: {internal_cluster_dir}")

    # -------------------------
    # 3) Mirror internal results into sketch/partfield_test
    # -------------------------
    external_feat_dir = OUT_ROOT / "partfield_features"
    external_cluster_dir = OUT_ROOT / "partfield_clusters_raw"

    print("\n[INFO] Copying PartField results into sketch/partfield_test/ ...")
    copy_dir(internal_feat_root, external_feat_dir)
    copy_dir(internal_cluster_dir, external_cluster_dir)
    print(f"[INFO] Features copied to: {external_feat_dir}")
    print(f"[INFO] Clusters copied to:  {external_cluster_dir}")

    # -------------------------
    # 4) Export CLEAN K=20 labels into sketch/partfield_test
    # -------------------------
    out_k20 = OUT_ROOT / f"{stem}_k20.npy"
    export_k20_for_single_model(internal_cluster_dir, stem=stem, out_npy_path=out_k20)

    print("\n🎯 Final K=20 label file saved:")
    print(out_k20)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
