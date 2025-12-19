#!/usr/bin/env python3
"""
5_run_partfield_segmentation.py

Run PartField on your (already decimated) mesh:
  sketch/3d/trellis_shape.glb

Do ONLY clustering for 20 clusters and save TWO debug outputs next to the mesh:
  - sketch/3d/clustering_k20.npy
  - sketch/3d/clustering_k20.glb

Assumes your repo layout:
  whole_pipeline/
    packages/PartField/          (repo)
    sketch/3d/trellis_shape.glb  (mesh)

Run inside `partfield` conda env:
  cd ~/Desktop/sketch_lib/whole_pipeline
  python 5_run_partfield_segmentation.py
"""

import sys
import subprocess
from pathlib import Path

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# -------------------------
# PATHS
# -------------------------
ROOT = Path(__file__).resolve().parent
PARTFIELD_ROOT = ROOT / "packages" / "PartField"

MESH_GLB = ROOT / "sketch" / "3d" / "trellis_shape.glb"

# REQUIRED OUTPUTS (as you asked)
OUT_DIR = MESH_GLB.parent
OUT_LABELS = OUT_DIR / "clustering_k20.npy"
OUT_GLB = OUT_DIR / "clustering_k20.glb"

# PartField config/ckpt are relative to PARTFIELD_ROOT
CFG_PATH = "configs/final/demo.yaml"
CKPT_PATH = "model/model_objaverse.ckpt"

# Keep names unique to avoid collisions across runs
RESULT_NAME = f"partfield_features/{MESH_GLB.stem}_mesh"
CLUSTER_DUMP_NAME = f"clustering/{MESH_GLB.stem}_k20"

TARGET_K = 20


def _run(cmd, cwd: Path):
    print("\n[CMD]", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=str(cwd), check=True)


def _select_best_k(labels_path_candidates, k: int):
    best_path = None
    best_diff = None

    for p in labels_path_candidates:
        try:
            lab = np.load(p).reshape(-1)
        except Exception:
            continue
        uniq = np.unique(lab[lab >= 0])
        diff = abs(len(uniq) - k)
        if best_path is None or diff < best_diff:
            best_path = p
            best_diff = diff
        if diff == 0:
            break

    if best_path is None:
        raise RuntimeError("Could not load any valid cluster label file.")
    return best_path


def _export_colored_mesh_glb(mesh_glb_path: Path, point_labels: np.ndarray, out_glb_path: Path):
    """
    PartField clustering labels are per 'feature point' (P), not per mesh vertex (V).
    For debugging, we:
      1) load mesh vertices/faces
      2) sample P points on surface
      3) map each vertex to nearest sampled point -> vertex label
      4) export GLB with per-vertex colors
    """
    tm = trimesh.load(mesh_glb_path, force="mesh")
    if isinstance(tm, trimesh.Scene):
        # take first geometry if scene
        geoms = list(tm.geometry.values())
        if not geoms:
            raise RuntimeError(f"Loaded a Scene but found no geometry in: {mesh_glb_path}")
        tm = geoms[0]

    V = np.asarray(tm.vertices, dtype=np.float32)
    F = np.asarray(tm.faces, dtype=np.int64)

    P = int(point_labels.shape[0])
    if P <= 0:
        raise RuntimeError("Empty labels array; cannot export GLB.")

    # sample P points on the mesh surface (for correspondence)
    points_xyz, _ = trimesh.sample.sample_surface(tm, P)
    points_xyz = points_xyz.astype(np.float32)

    # nearest neighbor: vertex -> sampled point -> label
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(points_xyz)
    _, idx = nn.kneighbors(V)
    idx = idx.reshape(-1)

    point_labels = np.squeeze(point_labels).astype(np.int64)
    vertex_labels = point_labels[idx]  # (V,)

    # assign colors
    unique_labels = np.unique(vertex_labels)
    colormap = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_color = {
        lab: np.array(colormap(i)[:3], dtype=np.float32)
        for i, lab in enumerate(unique_labels)
    }
    colors = np.stack([label_to_color[l] for l in vertex_labels], axis=0)  # (V,3) in [0,1]

    out_mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    out_mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)

    out_mesh.export(out_glb_path)
    print(f"[SAVE] {out_glb_path}")


def main():
    if not PARTFIELD_ROOT.is_dir():
        raise RuntimeError(f"Missing PartField folder: {PARTFIELD_ROOT}")
    if not MESH_GLB.is_file():
        raise RuntimeError(f"Missing mesh: {MESH_GLB}")

    data_dir = MESH_GLB.parent.resolve()
    stem = MESH_GLB.stem

    print("[INFO] Mesh:", MESH_GLB)
    print("[INFO] Data dir:", data_dir)
    print("[INFO] Output labels:", OUT_LABELS)
    print("[INFO] Output GLB  :", OUT_GLB)
    print("[INFO] PartField root:", PARTFIELD_ROOT)

    # -------------------------
    # 1) Feature extraction (mesh mode)
    # -------------------------
    extract_cmd = [
        sys.executable, "partfield_inference.py",
        "-c", CFG_PATH,
        "--opts",
        "continue_ckpt", CKPT_PATH,
        "result_name", RESULT_NAME,
        "dataset.data_path", str(data_dir),  # absolute folder (PartField does os.listdir)
        "is_pc", "False",                    # mesh mode
    ]
    print("\n[PartField] Extracting features...")
    _run(extract_cmd, PARTFIELD_ROOT)

    internal_feat_root = PARTFIELD_ROOT / "exp_results" / RESULT_NAME
    if not internal_feat_root.exists():
        raise RuntimeError(f"Feature output not found: {internal_feat_root}")

    # -------------------------
    # 2) Clustering (max K=20)
    # -------------------------
    internal_cluster_dir = PARTFIELD_ROOT / "exp_results" / CLUSTER_DUMP_NAME
    cluster_cmd = [
        sys.executable, "run_part_clustering.py",
        "--root", str(internal_feat_root),
        "--dump_dir", str(internal_cluster_dir),
        "--source_dir", str(data_dir),
        "--max_num_clusters", str(TARGET_K),
        "--is_pc", "False",
    ]
    print("\n[PartField] Clustering parts (K<=20)...")
    _run(cluster_cmd, PARTFIELD_ROOT)

    # -------------------------
    # 3) Pick best K=20 label file and save as clustering_k20.npy
    # -------------------------
    candidates = list(internal_cluster_dir.rglob(f"*{stem}*.npy"))
    if not candidates:
        raise RuntimeError(f"No cluster label .npy produced under: {internal_cluster_dir}")

    best = _select_best_k(candidates, TARGET_K)
    labels = np.load(best).reshape(-1)
    uniq = np.unique(labels[labels >= 0])
    print(f"\n[SELECT] {best.name} -> {len(uniq)} clusters (target={TARGET_K})")

    np.save(OUT_LABELS, labels)
    print(f"[SAVE] {OUT_LABELS}")

    # -------------------------
    # 4) Export clustering_k20.glb (colored mesh for debugging)
    # -------------------------
    _export_colored_mesh_glb(MESH_GLB, labels, OUT_GLB)

    print("\n✅ PartField segmentation done.")


if __name__ == "__main__":
    main()
