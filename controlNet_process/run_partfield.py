#!/usr/bin/env python3
"""
run_partfield.py

Goal
-----
Run PartField (feature extraction + clustering), then save EVERYTHING needed to later
map clustering labels back onto the ORIGINAL input shape (TARGET_PLY) using kNN/NN.

What gets saved (all under: ROOT_DIR/sketch/partfield/)
-------------------------------------------------------
A) PartField-side (label space)
  1) clusters_k20.npy              (M,)   labels on PartField point set
  2) partfield_coords.npy          (M,3)  the point coordinates that clusters_k20 refers to
  3) fused_model_partfield_k20.ply (M pts) colored PartField point cloud (visual/debug)

B) Original-side mapping artifacts (target space)
  4) target_points.npy             (N,3)  original input points from TARGET_PLY (for reproducibility)
  5) target_to_partfield_nn_idx.npy (N,)  NN mapping: for each original point -> nearest PartField point index
  6) target_to_partfield_nn_dist.npy (N,) distances (sanity / thresholding)
  7) fused_model_segmented_k20.ply (N pts) colored ORIGINAL input point cloud (ready for your camera pipeline)

Notes
-----
- This script tries hard to extract the PartField point coordinates from PartField feature outputs.
- If your PartField repo DOES NOT save any coordinates at all (rare but possible), the script
  will print a debug inventory and then error, and you must patch PartField to dump coords.
"""

import sys
import subprocess
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

try:
    import torch
except Exception:
    torch = None

# -----------------------
# CONFIG
# -----------------------
ROOT_DIR = Path(__file__).resolve().parent
PARTFIELD_ROOT = ROOT_DIR / "PartField"

TARGET_PLY = ROOT_DIR / "sketch" / "3d_reconstruction" / "fused_model.ply"
TARGET_DIR = TARGET_PLY.parent

OUT_DIR = ROOT_DIR / "sketch" / "partfield"

QUERY_CHUNK = 250_000
DEBUG_COORDS_SCAN = True


# -----------------------
# Utils: array handling
# -----------------------
def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    try:
        return np.asarray(x)
    except Exception:
        return None


def _as_nx3(arr: np.ndarray) -> np.ndarray | None:
    """Try to coerce common layouts into (N,3)."""
    if not isinstance(arr, np.ndarray):
        return None
    if arr.size == 0 or not np.isfinite(arr).all():
        return None

    # (N,3)
    if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= 100:
        return arr

    # (3,N)
    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] >= 100:
        return arr.T

    # (B,N,3) -> take first batch
    if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[1] >= 100 and arr.shape[0] <= 16:
        return arr[0]

    # (N,1,3)
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[-1] == 3 and arr.shape[0] >= 100:
        return arr[:, 0, :]

    return None


def _extract_xyz_from_obj(obj):
    """Return (xyz, where_str) or (None, None)."""
    if isinstance(obj, np.ndarray):
        nx3 = _as_nx3(obj)
        return (nx3, "ndarray") if nx3 is not None else (None, None)

    if isinstance(obj, dict):
        key_priority = ["coords", "xyz", "points", "pc", "pos", "vertices", "verts", "pcl", "cloud"]
        for k in key_priority:
            if k in obj:
                arr = _to_numpy(obj[k])
                nx3 = _as_nx3(arr) if arr is not None else None
                if nx3 is not None:
                    return nx3, f"dict['{k}']"

        for k, v in obj.items():
            if isinstance(v, (dict, list, tuple)):
                nx3, where = _extract_xyz_from_obj(v)
                if nx3 is not None:
                    return nx3, f"dict['{k}'] -> {where}"
            else:
                arr = _to_numpy(v)
                nx3 = _as_nx3(arr) if arr is not None else None
                if nx3 is not None:
                    return nx3, f"dict['{k}']"

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            nx3, where = _extract_xyz_from_obj(v)
            if nx3 is not None:
                return nx3, f"list[{i}] -> {where}"

    return None, None


def _quick_peek_file(p: Path):
    infos = []
    try:
        if p.suffix == ".npy":
            obj = np.load(p, allow_pickle=True)
            if isinstance(obj, np.ndarray):
                infos.append(("npy", obj.shape, str(obj.dtype)))
                nx3 = _as_nx3(obj)
                if nx3 is not None and nx3 is not obj:
                    infos.append(("npy->nx3", nx3.shape, str(nx3.dtype)))
            return infos

        if p.suffix == ".npz":
            z = np.load(p, allow_pickle=True)
            for k in z.files[:20]:
                arr = z[k]
                infos.append((f"npz['{k}']", arr.shape, str(arr.dtype)))
                nx3 = _as_nx3(arr)
                if nx3 is not None and nx3 is not arr:
                    infos.append((f"npz['{k}']->nx3", nx3.shape, str(nx3.dtype)))
            return infos

        if p.suffix in (".pt", ".pth"):
            if torch is None:
                return infos
            obj = torch.load(p, map_location="cpu")
            if isinstance(obj, dict):
                for k in list(obj.keys())[:30]:
                    arr = _to_numpy(obj[k])
                    if isinstance(arr, np.ndarray):
                        infos.append((f"torch['{k}']", arr.shape, str(arr.dtype)))
                        nx3 = _as_nx3(arr)
                        if nx3 is not None and nx3 is not arr:
                            infos.append((f"torch['{k}']->nx3", nx3.shape, str(nx3.dtype)))
            return infos
    except Exception:
        return infos

    return infos


# -----------------------
# PartField outputs: labels + coords extraction
# -----------------------
def export_clusters_k20(cluster_dir: Path, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = list(cluster_dir.rglob(f"*{stem}*.npy"))
    if not matches:
        raise RuntimeError(f"No cluster files found for stem='{stem}' under: {cluster_dir}")

    best_file, best_diff, best_k = None, float("inf"), -1
    for f in matches:
        try:
            l = np.load(f).reshape(-1)
            k = len(np.unique(l[l >= 0]))
            diff = abs(k - 20)
            if diff < best_diff or (diff == best_diff and k > best_k):
                best_file, best_diff, best_k = f, diff, k
        except Exception:
            continue

    if best_file is None:
        raise RuntimeError("Found cluster files but none could be loaded/validated.")

    labels = np.load(best_file).reshape(-1).astype(np.int64, copy=False)
    out_path = out_dir / "clusters_k20.npy"
    np.save(out_path, labels)
    print(f"[CLUSTERS] Using: {best_file} (K≈{best_k}, len={labels.shape[0]})")
    print(f"[CLUSTERS] Saved: {out_path}")
    return out_path


def extract_partfield_coords(feat_root: Path, target_stem: str, out_dir: Path, expected_len: int) -> Path:
    """
    Find an (M,3) point set inside PartField feature outputs with M == expected_len.
    Save it to out_dir/partfield_coords.npy and return that path.
    """
    candidates = []
    for ext in ("*.npy", "*.npz", "*.pt", "*.pth"):
        candidates.extend(feat_root.rglob(ext))
    if not candidates:
        raise RuntimeError(f"No candidate feature files found under: {feat_root}")

    def score(p: Path) -> int:
        s = str(p).lower()
        sc = 0
        if target_stem.lower() in s:
            sc += 100
        name = p.name.lower()
        if "coord" in name or "xyz" in name or "point" in name or "pc" in name or "pos" in name:
            sc += 20
        sc -= len(p.parts)
        return sc

    candidates.sort(key=score, reverse=True)

    if DEBUG_COORDS_SCAN:
        print("\n[COORDS][DEBUG] Top candidate files (peek shapes):")
        for p in candidates[:40]:
            infos = _quick_peek_file(p)
            if infos:
                print(f"  - {p}")
                for desc, sh, dt in infos[:6]:
                    print(f"      {desc:18s} shape={sh} dtype={dt}")

    for p in candidates[:500]:
        try:
            xyz, where = None, None

            if p.suffix == ".npy":
                obj = np.load(p, allow_pickle=True)
                xyz, where = _extract_xyz_from_obj(obj)

            elif p.suffix == ".npz":
                z = np.load(p, allow_pickle=True)
                for k in z.files:
                    nx3 = _as_nx3(z[k])
                    if nx3 is not None:
                        xyz, where = nx3, f"npz['{k}']"
                        break

            elif p.suffix in (".pt", ".pth"):
                if torch is None:
                    continue
                obj = torch.load(p, map_location="cpu")
                xyz, where = _extract_xyz_from_obj(obj)

            if xyz is None:
                continue

            xyz = np.asarray(xyz, dtype=np.float32)
            if xyz.shape[0] != expected_len:
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "partfield_coords.npy"
            np.save(out_path, xyz)
            print(f"\n[COORDS] Extracted PartField coords from: {p}")
            print(f"[COORDS] Found at: {where}")
            print(f"[COORDS] Saved: {out_path}  (shape={xyz.shape})")
            return out_path

        except Exception:
            continue

    raise RuntimeError(
        "Failed to extract PartField coords with matching length.\n"
        f"Expected coords length = {expected_len}\n"
        f"Searched under: {feat_root}\n"
        "Recommended fix: patch PartField to dump coords during inference (coords.npy)."
    )


# -----------------------
# Saving visuals + mapping
# -----------------------
def save_colored_partfield_ply(coords_path: Path, labels_path: Path, out_dir: Path) -> Path:
    """Save a colored point cloud in PartField point space for debugging."""
    coords = np.load(coords_path).astype(np.float32)
    labels = np.load(labels_path).reshape(-1).astype(np.int64)
    if coords.shape[0] != labels.shape[0]:
        raise RuntimeError("partfield_coords and clusters_k20 length mismatch.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    colors = cmap((labels % 20) / 20.0)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_ply = out_dir / "fused_model_partfield_k20.ply"
    o3d.io.write_point_cloud(str(out_ply), pcd)
    print(f"[OUT] Colored PartField-space PLY: {out_ply}")
    return out_ply


def map_labels_to_original_and_save(
    target_ply: Path,
    coords_path: Path,
    labels_path: Path,
    out_dir: Path,
    query_chunk: int = QUERY_CHUNK,
):
    """
    NN-map ORIGINAL points -> PartField coords, then color ORIGINAL point cloud.
    Save mapping arrays so later you can do kNN voting as well if you want.
    """
    partfield_coords = np.load(coords_path).astype(np.float32)
    partfield_labels = np.load(labels_path).reshape(-1).astype(np.int64)
    if partfield_coords.shape[0] != partfield_labels.shape[0]:
        raise RuntimeError("partfield_coords and clusters_k20 length mismatch.")

    pcd = o3d.io.read_point_cloud(str(target_ply))
    target_points = np.asarray(pcd.points).astype(np.float32)
    if target_points.shape[0] == 0:
        raise RuntimeError(f"Empty target ply: {target_ply}")

    print(f"[MAP] Original points: {target_points.shape[0]}")
    print(f"[MAP] PartField coords: {partfield_coords.shape[0]}")

    tree = cKDTree(partfield_coords)

    nn_idx = np.empty((target_points.shape[0],), dtype=np.int64)
    nn_dist = np.empty((target_points.shape[0],), dtype=np.float32)

    print("[MAP] KD-tree NN query (original -> PartField coords)...")
    n = target_points.shape[0]
    for s in range(0, n, query_chunk):
        e = min(n, s + query_chunk)
        d, idx = tree.query(target_points[s:e], workers=-1)
        nn_idx[s:e] = idx.astype(np.int64, copy=False)
        nn_dist[s:e] = d.astype(np.float32, copy=False)

    mapped_labels = partfield_labels[nn_idx]

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    colors = cmap((mapped_labels % 20) / 20.0)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "target_points.npy", target_points)
    np.save(out_dir / "target_to_partfield_nn_idx.npy", nn_idx)
    np.save(out_dir / "target_to_partfield_nn_dist.npy", nn_dist)

    out_ply = out_dir / "fused_model_segmented_k20.ply"
    o3d.io.write_point_cloud(str(out_ply), pcd)

    print(
        f"[MAP] NN dist: mean={float(nn_dist.mean()):.6f}, "
        f"median={float(np.median(nn_dist)):.6f}, max={float(nn_dist.max()):.6f}"
    )
    print(f"[OUT] Segmented ORIGINAL-space PLY: {out_ply}")
    return out_ply


# -----------------------
# Main
# -----------------------
def main():
    if not PARTFIELD_ROOT.is_dir():
        raise RuntimeError(f"PartField repo not found at: {PARTFIELD_ROOT}")
    if not TARGET_PLY.exists():
        raise RuntimeError(f"TARGET_PLY not found: {TARGET_PLY}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Feature extraction
    extract_cmd = [
        sys.executable, "partfield_inference.py",
        "-c", "configs/final/demo.yaml",
        "--opts",
        "continue_ckpt", "model/model_objaverse.ckpt",
        "result_name", "partfield_features/single_inference",
        "dataset.data_path", str(TARGET_DIR),
        "is_pc", "True",
    ]
    print("\n[1/3] PartField extracting features...")
    subprocess.run(extract_cmd, cwd=PARTFIELD_ROOT, check=True)

    # 2) Clustering
    feat_root = PARTFIELD_ROOT / "exp_results" / "partfield_features" / "single_inference"
    cluster_dir = PARTFIELD_ROOT / "exp_results" / "clustering" / "single_inference"

    cluster_cmd = [
        sys.executable, "run_part_clustering.py",
        "--root", str(feat_root),
        "--dump_dir", str(cluster_dir),
        "--source_dir", str(TARGET_DIR),
        "--max_num_clusters", "20",
        "--is_pc", "True",
    ]
    print("\n[2/3] PartField clustering parts...")
    subprocess.run(cluster_cmd, cwd=PARTFIELD_ROOT, check=True)

    # 3) Export labels
    print(f"\n[3/3] Saving everything into: {OUT_DIR}")
    labels_path = export_clusters_k20(cluster_dir, OUT_DIR, stem=TARGET_PLY.stem)
    labels = np.load(labels_path).reshape(-1)

    # 4) Extract coords for labels (M must match len(labels))
    coords_path = extract_partfield_coords(
        feat_root=feat_root,
        target_stem=TARGET_PLY.stem,
        out_dir=OUT_DIR,
        expected_len=int(labels.shape[0]),
    )

    # 5) Save PartField-space colored point cloud (debug)
    save_colored_partfield_ply(coords_path, labels_path, OUT_DIR)

    # 6) Map labels to ORIGINAL input points + save mapping + segmented original ply
    map_labels_to_original_and_save(
        target_ply=TARGET_PLY,
        coords_path=coords_path,
        labels_path=labels_path,
        out_dir=OUT_DIR,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
