#!/usr/bin/env python3
"""
trellis_to3d.py

Run TRELLIS image-to-3D on a single image and export a mesh.

Assumes directory layout:

    real_3D/
      TRELLIS/              # git repo (contains trellis package)
      TRELLIS-image-large/  # downloaded HF weights
      trellis_to3d.py
      0.png                 # your sketch

Usage (inside `trellis` conda env):

    cd ~/Desktop/sketch_lib/real_3D
    python trellis_to3d.py      # uses 0.png by default
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
TRELLIS_REPO = ROOT / "TRELLIS"          # repo
MODEL_DIR = ROOT / "TRELLIS-image-large" # local weights
IMG_PATH = ROOT / "0.png"                # default input

if not TRELLIS_REPO.is_dir():
    raise RuntimeError(
        f"Cannot find TRELLIS repo at {TRELLIS_REPO}.\n"
        "Make sure you have:\n"
        "  git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git TRELLIS"
    )

if not MODEL_DIR.is_dir():
    raise RuntimeError(
        f"Cannot find local model dir at {MODEL_DIR}.\n"
        "Download it with:\n"
        "  huggingface-cli download microsoft/TRELLIS-image-large \\\n"
        "      --local-dir TRELLIS-image-large --local-dir-use-symlinks False"
    )

if not IMG_PATH.is_file():
    raise FileNotFoundError(f"Input image not found: {IMG_PATH}")

# Make repo importable as `trellis`
sys.path.insert(0, str(TRELLIS_REPO))

# ---------------------------------------------------------------------
# Backend config: avoid flash_attn, use xformers instead
# ---------------------------------------------------------------------
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

from PIL import Image
import torch
import trimesh

from trellis.pipelines import TrellisImageTo3DPipeline


def main():
    print(f"[info] ROOT         : {ROOT}")
    print(f"[info] TRELLIS repo : {TRELLIS_REPO}")
    print(f"[info] Model dir    : {MODEL_DIR}")
    print(f"[info] Input image  : {IMG_PATH}")

    image = Image.open(IMG_PATH).convert("RGB")

    print("[info] Loading Trellis pipeline from local folder...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(MODEL_DIR))

    if torch.cuda.is_available():
        pipeline.cuda()
        print(f"[info] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        pipeline.cpu()
        print("[warn] CUDA not available, using CPU (will be slow).")

    # ---------------- run TRELLIS ----------------
    print("[info] Running image-to-3D...")
    outputs = pipeline.run(
        image,
        seed=1,
    )

    # ---------------- export raw mesh ----------------
    mesh = outputs["mesh"][0]  # MeshExtractResult

    vertices = mesh.vertices.detach().cpu().numpy()  # (N, 3)
    faces = mesh.faces.detach().cpu().numpy()        # (M, 3)

    out_dir = ROOT / "trellis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = out_dir / "0_trellis_raw.glb"
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tm.export(mesh_path)
    print(f"[info] Saved raw mesh to {mesh_path}")

    # Optional: also save Gaussian as PLY (no nvdiffrast involved)
    gauss_path = out_dir / "0_trellis_gaussian.ply"
    outputs["gaussian"][0].save_ply(str(gauss_path))
    print(f"[info] Saved Gaussian PLY to {gauss_path}")


if __name__ == "__main__":
    main()
