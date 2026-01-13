#!/usr/bin/env python3
"""
map_partfield_to_input.py

Goal:
- You already have PartField cluster labels (K=20) produced on a *PartField-processed point set*
  (often re-sampled / normalized / re-ordered) -> labels file like:
      sketch/partfield_test/fused_model_k20.npy

- You want to map those labels BACK onto your original input point cloud:
      sketch/3d_reconstruction/fused_model.ply

Outputs (all under sketch/partfield_test/):
  - fused_model_k20_on_input.npy          # (N_input,) int labels for original points
  - fused_model_pf_nn_index.npy           # (N_input,) nearest PF point index
  - fused_model_pf_nn_dist.npy            # (N_input,) nearest neighbor distance
  - fused_model_k20_on_input_colored.ply  # colored PLY (for quick visualization)
  - mapping_report.json                   # summary stats

How it works:
1) Load the original input PLY points (input space).
2) Load the PartField point cloud that the labels correspond to (PF space).
   - We auto-discover it by scanning sketch/partfield_test/partfield_clusters_raw/ and
     selecting a .ply whose point count best matches len(labels).
   - If none found, we fall back to sketch/partfield_test/partfield_input/fused_model.ply
     (but if PF truly changed the shape, you WANT the discovered PF .ply).
3) Build a KD-tree on PF points.
4) For each input point: find nearest PF point -> transfer that PF label.

Notes about "camera pos, point id, etc":
- This script does NOT destroy your original metadata because it never reorders your input points.
- It outputs label arrays aligned with the original input point order.
- If your downstream code uses "point id" as the row index into the original point array, you're good.

Dependencies:
- numpy required
- open3d OR scipy is used for KDTree + PLY I/O.
  - If scipy is available, we use scipy.spatial.cKDTree (fast).
  - Otherwise we use open3d KDTreeFlann.
"""

import json
from pathlib import Path
import numpy as np


def _load_points_ply(path: Path) -> np.ndarray:
    """
    Load PLY points as (N,3) float32.
    Prefer open3d (common in your stack).
    """
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(
            "open3d is required to load .ply in this script.\n"
            "Install with: pip install open3d\n"
            f"Import error: {e}"
        )

    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError(f"Bad point shape from {path}: {pts.shape}")
    return pts


def _save_colored_ply(path: Path, pts: np.ndarray, labels: np.ndarray, k_hint: int = 20) -> None:
    """
    Save a colored PLY for visualization.
    Colors are deterministic per label (hash-based).
    label == -1 -> black
    """
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(
            "open3d is required to save colored .ply in this script.\n"
            "Install with: pip install open3d\n"
            f"Import error: {e}"
        )

    pts = np.asarray(pts, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    if pts.shape[0] != labels.shape[0]:
        raise ValueError(f"pts and labels length mismatch: {pts.shape[0]} vs {labels.shape[0]}")

    def label_to_rgb01(l: int) -> np.ndarray:
        if l < 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # deterministic pseudo-random color from label
        x = (l + 1) * 2654435761  # Knuth multiplicative hash
        r = ((x >> 0) & 255) / 255.0
        g = ((x >> 8) & 255) / 255.0
        b = ((x >> 16) & 255) / 255.0
        # make colors a bit brighter / less muddy
        rgb = np.array([r, g, b], dtype=np.float32)
        rgb = 0.25 + 0.75 * rgb
        return np.clip(rgb, 0.0, 1.0)

    colors = np.stack([label_to_rgb01(int(l)) for l in labels], axis=0).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)
    if not ok:
        raise RuntimeError(f"Failed to write ply: {path}")


def _find_best_pf_ply(partfield_test_dir: Path, labels_len: int) -> Path | None:
    """
    Try to locate the actual PF point cloud that the labels correspond to.
    We search in:
      sketch/partfield_test/partfield_clusters_raw/**.ply
      sketch/partfield_test/partfield_features/**.ply
      sketch/partfield_test/**.ply (fallback)

    We pick the .ply whose point count is closest to labels_len.
    """
    search_roots = [
        partfield_test_dir / "partfield_clusters_raw",
        partfield_test_dir / "partfield_features",
        partfield_test_dir,
    ]

    cands: list[Path] = []
    for r in search_roots:
        if r.exists():
            cands.extend(list(r.rglob("*.ply")))

    # Remove the obvious input copies (still keep as fallback candidate though)
    # We'll just score by point count; if PF produced a different one, it should win.

    best_path = None
    best_diff = None
    best_n = None

    for p in cands:
        try:
            pts = _load_points_ply(p)
            n = int(pts.shape[0])
        except Exception:
            continue

        diff = abs(n - labels_len)
        if best_path is None or diff < best_diff:
            best_path = p
            best_diff = diff
            best_n = n

    if best_path is not None:
        print(f"[INFO] Auto-selected PF point cloud: {best_path} (N={best_n}, labels={labels_len}, diff={best_diff})")
    return best_path


def _nn_map_scipy(pf_pts: np.ndarray, in_pts: np.ndarray):
    from scipy.spatial import cKDTree
    tree = cKDTree(pf_pts.astype(np.float64))
    dists, idx = tree.query(in_pts.astype(np.float64), k=1, workers=-1)
    return idx.astype(np.int64), dists.astype(np.float32)


def _nn_map_open3d(pf_pts: np.ndarray, in_pts: np.ndarray):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pf_pts.astype(np.float64))
    kdt = o3d.geometry.KDTreeFlann(pcd)

    idx = np.empty((in_pts.shape[0],), dtype=np.int64)
    dists = np.empty((in_pts.shape[0],), dtype=np.float32)

    for i in range(in_pts.shape[0]):
        q = in_pts[i].astype(np.float64)
        k, ind, dist2 = kdt.search_knn_vector_3d(q, 1)
        if k <= 0:
            idx[i] = -1
            dists[i] = np.inf
        else:
            idx[i] = int(ind[0])
            dists[i] = float(np.sqrt(dist2[0])) if len(dist2) > 0 else 0.0
    return idx, dists


def main():
    ROOT = Path(__file__).resolve().parent

    # -------------------------
    # Inputs (relative paths)
    # -------------------------
    INPUT_PLY = ROOT / "sketch" / "3d_reconstruction" / "fused_model.ply"
    TEST_DIR = ROOT / "sketch" / "partfield_test"
    LABELS_NPY = TEST_DIR / "fused_model_k20.npy"  # produced by your PartField run

    # Fallback PF input copy (often identical to INPUT_PLY, but kept here)
    PF_INPUT_COPY = TEST_DIR / "partfield_input" / "fused_model.ply"

    if not INPUT_PLY.is_file():
        raise RuntimeError(f"Missing original input ply: {INPUT_PLY}")
    if not LABELS_NPY.is_file():
        raise RuntimeError(f"Missing labels npy: {LABELS_NPY}")

    labels_pf = np.load(LABELS_NPY).reshape(-1)
    print(f"[INFO] Loaded PF labels: {LABELS_NPY} shape={labels_pf.shape}")

    # -------------------------
    # Load original input points
    # -------------------------
    in_pts = _load_points_ply(INPUT_PLY)
    print(f"[INFO] Loaded input points: {INPUT_PLY} N={in_pts.shape[0]}")

    # -------------------------
    # Find PF point cloud that labels correspond to
    # -------------------------
    pf_ply = _find_best_pf_ply(TEST_DIR, labels_len=int(labels_pf.shape[0]))
    if pf_ply is None:
        if PF_INPUT_COPY.is_file():
            pf_ply = PF_INPUT_COPY
            print(f"[WARN] Could not discover PF .ply; falling back to: {pf_ply}")
        else:
            raise RuntimeError(
                "Could not discover a PF .ply file to match labels, and fallback PF input copy is missing.\n"
                "Please check sketch/partfield_test/partfield_clusters_raw for any .ply outputs."
            )

    pf_pts = _load_points_ply(pf_ply)
    print(f"[INFO] Loaded PF points: {pf_ply} N={pf_pts.shape[0]}")

    if pf_pts.shape[0] != labels_pf.shape[0]:
        print(
            "[WARN] PF points count != labels length.\n"
            f"       PF points: {pf_pts.shape[0]}, labels: {labels_pf.shape[0]}\n"
            "       Mapping will proceed using nearest neighbor, but verify PF point source selection."
        )

    # -------------------------
    # Nearest-neighbor mapping (input -> PF)
    # -------------------------
    print("[INFO] Building NN mapping: input_points -> pf_points ...")
    try:
        import scipy  # noqa: F401
        nn_idx, nn_dist = _nn_map_scipy(pf_pts, in_pts)
        print("[INFO] Used scipy.spatial.cKDTree")
    except Exception:
        nn_idx, nn_dist = _nn_map_open3d(pf_pts, in_pts)
        print("[INFO] Used open3d.geometry.KDTreeFlann")

    # -------------------------
    # Transfer labels
    # -------------------------
    labels_on_input = np.full((in_pts.shape[0],), -1, dtype=np.int32)
    valid = (nn_idx >= 0) & (nn_idx < labels_pf.shape[0])
    labels_on_input[valid] = labels_pf[nn_idx[valid]].astype(np.int32)

    # Optional: you can enforce a distance cutoff here if you want.
    # Example:
    # cutoff = 0.01  # adjust to your scale
    # labels_on_input[nn_dist > cutoff] = -1

    # -------------------------
    # Save outputs
    # -------------------------
    out_labels = TEST_DIR / "fused_model_k20_on_input.npy"
    out_nn_idx = TEST_DIR / "fused_model_pf_nn_index.npy"
    out_nn_dist = TEST_DIR / "fused_model_pf_nn_dist.npy"
    out_colored = TEST_DIR / "fused_model_k20_on_input_colored.ply"
    out_report = TEST_DIR / "mapping_report.json"

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out_labels, labels_on_input)
    np.save(out_nn_idx, nn_idx.astype(np.int64))
    np.save(out_nn_dist, nn_dist.astype(np.float32))

    _save_colored_ply(out_colored, in_pts, labels_on_input)

    # Report
    uniq = np.unique(labels_on_input[labels_on_input >= 0])
    counts = {int(k): int((labels_on_input == k).sum()) for k in uniq}
    report = {
        "input_ply": str(INPUT_PLY),
        "pf_ply_used_for_mapping": str(pf_ply),
        "labels_file": str(LABELS_NPY),
        "n_input_points": int(in_pts.shape[0]),
        "n_pf_points": int(pf_pts.shape[0]),
        "n_labels_pf": int(labels_pf.shape[0]),
        "n_mapped_valid": int(valid.sum()),
        "nn_dist_min": float(np.min(nn_dist)),
        "nn_dist_mean": float(np.mean(nn_dist)),
        "nn_dist_p95": float(np.quantile(nn_dist, 0.95)),
        "nn_dist_max": float(np.max(nn_dist)),
        "unique_labels_on_input": [int(x) for x in uniq.tolist()],
        "label_counts_on_input": counts,
        "notes": [
            "labels_on_input is aligned with the original fused_model.ply point order.",
            "If your 'point id' is the index in the original point array, you can directly join metadata with labels_on_input.",
            "If mapping quality is poor, inspect nn_dist stats and consider adding a distance cutoff."
        ],
    }
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Saved mapped segmentation + files to:")
    print(f"  - {out_labels}")
    print(f"  - {out_nn_idx}")
    print(f"  - {out_nn_dist}")
    print(f"  - {out_colored}")
    print(f"  - {out_report}")


if __name__ == "__main__":
    main()
