#!/usr/bin/env python3
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from collections import defaultdict, Counter
from pathlib import Path

# ---------------------------------------------------------------------
# PATHS (EDIT IF NEEDED)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

VGGT_PLY = ROOT / "sketch" / "3d_reconstruction" / "fused_model.ply"
PARTFIELD_COORDS = (
    ROOT / "PartField"
    / "exp_results"
    / "partfield_features"
    / "single_inference"
    / "fused_model"
    / "coords.npy"
)

PARTFIELD_LABELS = ROOT / "sketch" / "3d_reconstruction" / "clusters_k20.npy"

OUT_LABELS = ROOT / "sketch" / "3d_reconstruction" / "vggt_clusters_k20.npy"
OUT_PLY = ROOT / "sketch" / "3d_reconstruction" / "vggt_partfield_k20.ply"

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("[LOAD] VGGT point cloud")
    pcd = o3d.io.read_point_cloud(str(VGGT_PLY))
    vggt_points = np.asarray(pcd.points)

    print("[LOAD] PartField coords + labels")
    pf_points = np.load(PARTFIELD_COORDS)
    pf_labels = np.load(PARTFIELD_LABELS).reshape(-1)

    assert len(pf_points) == len(pf_labels)

    print("[KD-TREE] Building VGGT tree")
    tree = cKDTree(vggt_points)

    print("[MAP] PartField → VGGT points")
    _, vggt_indices = tree.query(pf_points, workers=-1)

    # -----------------------------------------------------------------
    # Majority vote per VGGT point
    # -----------------------------------------------------------------
    votes = defaultdict(list)

    for pf_idx, v_idx in enumerate(vggt_indices):
        lbl = pf_labels[pf_idx]
        if lbl >= 0:
            votes[v_idx].append(lbl)

    vggt_labels = np.full(len(vggt_points), -1, dtype=np.int32)

    for v_idx, lbls in votes.items():
        vggt_labels[v_idx] = Counter(lbls).most_common(1)[0][0]

    print("[SAVE] VGGT cluster labels")
    np.save(OUT_LABELS, vggt_labels)

    # -----------------------------------------------------------------
    # Optional: colored visualization
    # -----------------------------------------------------------------
    print("[SAVE] Colored VGGT PLY")
    colors = np.zeros((len(vggt_points), 3), dtype=np.float64)

    for k in np.unique(vggt_labels):
        if k < 0:
            continue
        mask = vggt_labels == k
        colors[mask] = np.random.rand(3)

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(OUT_PLY), pcd)

    print("✅ DONE")
    print(f"Labels: {OUT_LABELS}")
    print(f"PLY:    {OUT_PLY}")

if __name__ == "__main__":
    main()
