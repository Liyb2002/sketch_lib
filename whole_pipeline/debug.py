#!/usr/bin/env python3
"""
6_vis_partfield_segmentation.py

Visualize PartField K=20 segmentation outputs saved by 5_run_partfield_segmentation.py:

  sketch/3d/clustering_k20.glb   (colored mesh)
  sketch/3d/clustering_k20.npy   (labels, for stats)

Shows:
- an interactive Open3D window (if available)
- prints basic label stats (cluster counts)

Run (any env with open3d + numpy):
  cd ~/Desktop/sketch_lib/whole_pipeline
  python 6_vis_partfield_segmentation.py
"""

from pathlib import Path
import numpy as np

try:
    import open3d as o3d
except Exception as e:
    o3d = None

ROOT = Path(__file__).resolve().parent
GLB_PATH = ROOT / "sketch" / "3d" / "clustering_k20.glb"
NPY_PATH = ROOT / "sketch" / "3d" / "clustering_k20.npy"


def print_label_stats(labels: np.ndarray):
    labels = labels.reshape(-1)
    valid = labels[labels >= 0]
    uniq, counts = np.unique(valid, return_counts=True)
    order = np.argsort(-counts)

    print("\n[stats] labels:")
    print("  total:", labels.shape[0])
    print("  valid:", valid.shape[0])
    print("  num_clusters:", uniq.shape[0])

    print("\n[stats] top clusters by size:")
    for i in order[:10]:
        print(f"  label {int(uniq[i]):2d}: {int(counts[i])} pts")


def vis_glb_open3d(glb_path: Path):
    if o3d is None:
        print("[WARN] open3d import failed. Install with: pip install open3d")
        return

    if not glb_path.exists():
        raise FileNotFoundError(f"Missing GLB: {glb_path}")

    mesh = o3d.io.read_triangle_mesh(str(glb_path), enable_post_processing=True)
    if mesh.is_empty():
        raise RuntimeError("Open3D loaded an empty mesh (GLB may be invalid).")

    # Ensure normals exist for lighting
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    print("\n[open3d] Controls:")
    print("  - Left drag: rotate")
    print("  - Shift+Left drag: translate")
    print("  - Scroll: zoom")
    print("  - Press 'H' in window for help")

    o3d.visualization.draw_geometries(
        [mesh],
        window_name="PartField Segmentation (clustering_k20.glb)",
        width=1280,
        height=800,
        mesh_show_back_face=True,
    )


def main():
    print("[info] GLB:", GLB_PATH)
    print("[info] NPY:", NPY_PATH)

    if NPY_PATH.exists():
        labels = np.load(NPY_PATH)
        print_label_stats(labels)
    else:
        print("[WARN] Missing labels npy; skipping stats.")

    vis_glb_open3d(GLB_PATH)


if __name__ == "__main__":
    main()
