#!/usr/bin/env python3
"""
run_partfield_single_and_map.py

MERGED from:
- run_partfield_single.py
- map_partfield_to_input.py

All paths are RELATIVE to this script's folder (ROOT).

- PartField package: ROOT/packages/PartField
- Input point cloud: ROOT/sketch/3d_reconstruction/fused_model.ply
- Output folder:     ROOT/sketch/partfield_test/

Existing outputs preserved exactly as before (under sketch/partfield_test/):
  1) Clean K≈20 labels (PF space):
     ROOT/sketch/partfield_test/fused_model_k20.npy

  2) Mirrored PartField internal outputs:
     ROOT/sketch/partfield_test/partfield_features/
     ROOT/sketch/partfield_test/partfield_clusters_raw/

  3) A local dataset folder used as PartField input:
     ROOT/sketch/partfield_test/partfield_input/fused_model.ply

  4) Mapping outputs (under sketch/partfield_test/):
     ROOT/sketch/partfield_test/fused_model_k20_on_input.npy
     ROOT/sketch/partfield_test/fused_model_pf_nn_index.npy
     ROOT/sketch/partfield_test/fused_model_pf_nn_dist.npy
     ROOT/sketch/partfield_test/fused_model_k20_on_input_colored.ply
     ROOT/sketch/partfield_test/mapping_report.json

NEW (as requested): also write the final mapped outputs into ROOT/sketch/3d_reconstruction/:
  - ROOT/sketch/3d_reconstruction/clustering_k20_points.npy
  - ROOT/sketch/3d_reconstruction/clustering_k20_points.ply

No other behavior changes.
"""

import sys
import subprocess
from pathlib import Path
import shutil
import json
import numpy as np


# -----------------------------------------------------------------------------
# Helpers from run_partfield_single.py
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Helpers from map_partfield_to_input.py
# -----------------------------------------------------------------------------
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


def _save_colored_ply(path: Path, pts: np.ndarray, labels: np.ndarray) -> None:
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
        x = (l + 1) * 2654435761  # Knuth multiplicative hash
        r = ((x >> 0) & 255) / 255.0
        g = ((x >> 8) & 255) / 255.0
        b = ((x >> 16) & 255) / 255.0
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ROOT = Path(__file__).resolve().parent

    # -------------------------
    # Relative paths (unchanged)
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
    if input_ply.resolve() != IN_PLY.resolve():
        shutil.copy2(IN_PLY, input_ply)

    stem = input_ply.stem  # fused_model
    print(f"[INFO] PartField dataset dir: {input_dir}")
    print(f"[INFO] Dataset file: {input_ply}")

    # -------------------------
    # PartField config (unchanged)
    # -------------------------
    cfg_path = "configs/final/demo.yaml"
    ckpt_path = "model/model_objaverse.ckpt"

    result_name = "partfield_features/partfield_test_single"
    clustering_name = "partfield_test_single"

    # -------------------------
    # 1) Feature extraction (unchanged)
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
    # 2) Clustering (unchanged)
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
    # 3) Mirror internal results into sketch/partfield_test (unchanged)
    # -------------------------
    external_feat_dir = OUT_ROOT / "partfield_features"
    external_cluster_dir = OUT_ROOT / "partfield_clusters_raw"

    print("\n[INFO] Copying PartField results into sketch/partfield_test/ ...")
    copy_dir(internal_feat_root, external_feat_dir)
    copy_dir(internal_cluster_dir, external_cluster_dir)
    print(f"[INFO] Features copied to: {external_feat_dir}")
    print(f"[INFO] Clusters copied to:  {external_cluster_dir}")

    # -------------------------
    # 4) Export CLEAN K≈20 labels into sketch/partfield_test (unchanged)
    # -------------------------
    out_k20_pf = OUT_ROOT / f"{stem}_k20.npy"
    export_k20_for_single_model(internal_cluster_dir, stem=stem, out_npy_path=out_k20_pf)

    # -------------------------
    # 5) Map labels back onto original input (same logic as map_partfield_to_input.py)
    # -------------------------
    LABELS_NPY = out_k20_pf  # produced above
    PF_INPUT_COPY = OUT_ROOT / "partfield_input" / "fused_model.ply"

    labels_pf = np.load(LABELS_NPY).reshape(-1)
    print(f"\n[INFO] Loaded PF labels: {LABELS_NPY} shape={labels_pf.shape}")

    in_pts = _load_points_ply(IN_PLY)
    print(f"[INFO] Loaded input points: {IN_PLY} N={in_pts.shape[0]}")

    pf_ply = _find_best_pf_ply(OUT_ROOT, labels_len=int(labels_pf.shape[0]))
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

    print("[INFO] Building NN mapping: input_points -> pf_points ...")
    try:
        import scipy  # noqa: F401
        nn_idx, nn_dist = _nn_map_scipy(pf_pts, in_pts)
        print("[INFO] Used scipy.spatial.cKDTree")
    except Exception:
        nn_idx, nn_dist = _nn_map_open3d(pf_pts, in_pts)
        print("[INFO] Used open3d.geometry.KDTreeFlann")

    labels_on_input = np.full((in_pts.shape[0],), -1, dtype=np.int32)
    valid = (nn_idx >= 0) & (nn_idx < labels_pf.shape[0])
    labels_on_input[valid] = labels_pf[nn_idx[valid]].astype(np.int32)

    # Save mapping outputs under sketch/partfield_test (unchanged)
    out_labels = OUT_ROOT / "fused_model_k20_on_input.npy"
    out_nn_idx = OUT_ROOT / "fused_model_pf_nn_index.npy"
    out_nn_dist = OUT_ROOT / "fused_model_pf_nn_dist.npy"
    out_colored = OUT_ROOT / "fused_model_k20_on_input_colored.ply"
    out_report = OUT_ROOT / "mapping_report.json"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(out_labels, labels_on_input)
    np.save(out_nn_idx, nn_idx.astype(np.int64))
    np.save(out_nn_dist, nn_dist.astype(np.float32))
    _save_colored_ply(out_colored, in_pts, labels_on_input)

    uniq = np.unique(labels_on_input[labels_on_input >= 0])
    counts = {int(k): int((labels_on_input == k).sum()) for k in uniq}
    report = {
        "input_ply": str(IN_PLY),
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

    print("\n✅ Saved mapped segmentation + files to sketch/partfield_test/:")
    print(f"  - {out_labels}")
    print(f"  - {out_nn_idx}")
    print(f"  - {out_nn_dist}")
    print(f"  - {out_colored}")
    print(f"  - {out_report}")

    # -------------------------
    # 6) ALSO write requested outputs into sketch/3d_reconstruction (ONLY ADDITION)
    # -------------------------
    RECON_DIR = ROOT / "sketch" / "3d_reconstruction"
    out_k20_npy_recon = RECON_DIR / "clustering_k20_points.npy"
    out_k20_ply_recon = RECON_DIR / "clustering_k20_points.ply"

    np.save(out_k20_npy_recon, labels_on_input.astype(np.int32))
    _save_colored_ply(out_k20_ply_recon, in_pts, labels_on_input)

    print("\n🎯 Requested outputs saved to sketch/3d_reconstruction/:")
    print(f"  - {out_k20_npy_recon}")
    print(f"  - {out_k20_ply_recon}")
    print("\n✅ Done.")

    # -------------------------
    # 7) CLEANUP: remove sketch/partfield_test
    # -------------------------
    shutil.rmtree(OUT_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
