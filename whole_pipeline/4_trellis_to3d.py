#!/usr/bin/env python3
"""
trellis_to3d.py

Run TRELLIS image-to-3D on a single image and export a mesh.
NOW: the saved GLB is downsampled (decimated) to ~TARGET_TRIS faces.

Assumes directory layout:

    real_3D/
      packages/
        TRELLIS/
        TRELLIS-image-large/
      trellis_to3d.py
      sketch/
        input.png
        3d/

Usage (inside `trellis` conda env):
    cd ~/Desktop/sketch_lib/real_3D
    python trellis_to3d.py
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
TRELLIS_REPO = ROOT / "packages" / "TRELLIS"
MODEL_DIR = ROOT / "packages" / "TRELLIS-image-large"
IMG_PATH = ROOT / "sketch" / "input.png"

OUT_DIR = ROOT / "sketch" / "3d"
OUT_GLB = OUT_DIR / "trellis_shape.glb"           # will be DECIMATED
OUT_PLY = OUT_DIR / "trellis_shape.ply"           # gaussian splat ply (unchanged)

# NEW: also save a decimated mesh as PLY (handy fallback)
OUT_MESH_PLY = OUT_DIR / "trellis_shape_mesh_decimated.ply"

# NEW: decimation target
TARGET_TRIS = 5000

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
import numpy as np
import trimesh
import open3d as o3d

from trellis.pipelines import TrellisImageTo3DPipeline


def decimate_mesh(vertices: np.ndarray, faces: np.ndarray, target_tris: int):
    """
    Quadric decimation using Open3D.
    Returns (dec_vertices, dec_faces).
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

    # cleanup before decimation
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    tri0 = np.asarray(mesh.triangles).shape[0]
    v0 = np.asarray(mesh.vertices).shape[0]
    print(f"[decimate] before: verts={v0} tris={tri0}")

    if tri0 <= target_tris:
        print(f"[decimate] skip (already <= {target_tris} tris)")
        simp = mesh
    else:
        print(f"[decimate] simplifying to target_tris={target_tris} ...")
        simp = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_tris))

        # cleanup after decimation
        simp.remove_degenerate_triangles()
        simp.remove_duplicated_triangles()
        simp.remove_non_manifold_edges()

    tri1 = np.asarray(simp.triangles).shape[0]
    v1 = np.asarray(simp.vertices).shape[0]
    print(f"[decimate] after : verts={v1} tris={tri1}")

    dec_v = np.asarray(simp.vertices).astype(np.float32)
    dec_f = np.asarray(simp.triangles).astype(np.int64)
    return dec_v, dec_f, simp


def main():
    print(f"[info] ROOT         : {ROOT}")
    print(f"[info] TRELLIS repo : {TRELLIS_REPO}")
    print(f"[info] Model dir    : {MODEL_DIR}")
    print(f"[info] Input image  : {IMG_PATH}")
    print(f"[info] Output GLB   : {OUT_GLB}  (decimated)")
    print(f"[info] Output PLY   : {OUT_PLY}  (gaussian)")
    print(f"[info] Target tris  : {TARGET_TRIS}")

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

    # ---------------- export mesh (DECIMATED) ----------------
    mesh = outputs["mesh"][0]  # MeshExtractResult

    vertices = mesh.vertices.detach().cpu().numpy()  # (N, 3)
    faces = mesh.faces.detach().cpu().numpy()        # (M, 3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[info] Decimating mesh before saving...")
    dec_v, dec_f, o3d_mesh = decimate_mesh(vertices, faces, TARGET_TRIS)

    # Save decimated mesh as GLB
    tm = trimesh.Trimesh(vertices=dec_v, faces=dec_f, process=False)
    tm.export(OUT_GLB)
    print(f"[info] Saved DECIMATED mesh GLB to {OUT_GLB}")

    # Also save decimated mesh as PLY (optional fallback)
    o3d.io.write_triangle_mesh(str(OUT_MESH_PLY), o3d_mesh)
    print(f"[info] Saved DECIMATED mesh PLY to {OUT_MESH_PLY}")

    # Save the Gaussian as PLY (unchanged)
    outputs["gaussian"][0].save_ply(str(OUT_PLY))
    print(f"[info] Saved Gaussian PLY to {OUT_PLY}")


if __name__ == "__main__":
    main()
