#!/usr/bin/env python3
"""
heat_map.py

Per-label voxel heatmaps over the WHOLE shape.

For a target label L, voxelize the whole point cloud. For each voxel v:
  k(v)      = total points in v
  m_L(v)    = points in v whose semantic label == L
  heat_L(v) = m_L(v) / k(v)

Then color EVERY point by heat_L(voxel(point)) with a piecewise colormap:
  heat=1.0 -> red
  heat=0.5 -> green
  heat=0.0 -> black

Outputs:
  out_dir/
    heatmaps_summary.json
    heatmaps/<label>/heat_map_<label>.ply
    heatmaps/<label>/summary.json

Also pops Open3D windows (optional) to visualize each label.
"""

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# ---------------- IO helpers ----------------

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _sanitize_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_\-]+", "", name)
    return name or "unknown"

def _load_primitives(primitives_json_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    raw = _load_json(primitives_json_path)
    if isinstance(raw, dict) and "primitives" in raw:
        return raw, raw["primitives"]
    if isinstance(raw, list):
        return {"primitives": raw}, raw
    raise ValueError(f"Unexpected primitives JSON format: {primitives_json_path}")

def _cluster_to_label_map(primitives_json_path: str) -> Dict[int, str]:
    _, prims = _load_primitives(primitives_json_path)
    out: Dict[int, str] = {}
    for p in prims:
        cid = int(p.get("cluster_id", -1))
        lab = str(p.get("label", "unknown"))
        if cid not in out:
            out[cid] = lab
    return out

def _load_points_ply(ply_path: str) -> np.ndarray:
    if o3d is None:
        raise RuntimeError("open3d is required to load PLY. Please `pip install open3d`.")
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Bad point cloud shape from {ply_path}: {pts.shape}")
    return pts.astype(np.float32)

def _load_cluster_ids(cluster_ids_path: str, expected_n: int) -> np.ndarray:
    arr = np.load(cluster_ids_path)
    arr = np.asarray(arr).reshape(-1)
    if arr.shape[0] != expected_n:
        raise ValueError(
            f"cluster_ids length mismatch: {cluster_ids_path}\n"
            f"  cluster_ids: {arr.shape[0]}\n"
            f"  points     : {expected_n}"
        )
    return arr.astype(np.int64)


# ---------------- voxel helpers ----------------

def _auto_voxel_size(points: np.ndarray, target_bins: int = 160) -> float:
    """
    Smaller voxel size -> more bins across bounding box diagonal.

    target_bins=160 is a good default for finer heat variation.
    Increase to ~220 for even smaller voxels; decrease for smoother.
    """
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    if diag < 1e-9:
        return 1e-3
    return max(diag / float(target_bins), 1e-4)

def _voxel_inv(points: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      inv: (N,) voxel index per point
      uniq: (M,) unique voxel keys (structured dtype)
    """
    keys = np.floor(points / float(voxel_size)).astype(np.int64)  # (N,3)
    key_view = keys.view([("x", np.int64), ("y", np.int64), ("z", np.int64)]).reshape(-1)
    uniq, inv = np.unique(key_view, return_inverse=True)
    return inv.astype(np.int64), uniq


# ---------------- color map ----------------

def _color_heat_red_green_black(h: np.ndarray) -> np.ndarray:
    """
    Piecewise color map:
      h=0   -> black
      h=0.5 -> green
      h=1   -> red
    Linear interpolation:
      [0, 0.5]: black -> green
      [0.5, 1]: green -> red
    """
    h = np.clip(h.astype(np.float32), 0.0, 1.0)
    rgb = np.zeros((h.shape[0], 3), dtype=np.float32)

    lo = h <= 0.5
    hi = ~lo

    # 0..0.5 : black -> green
    t = np.zeros_like(h)
    t[lo] = (h[lo] / 0.5)  # 0..1
    rgb[lo, 1] = t[lo]     # green channel

    # 0.5..1 : green -> red
    t2 = np.zeros_like(h)
    t2[hi] = (h[hi] - 0.5) / 0.5  # 0..1
    rgb[hi, 0] = t2[hi]           # red rises
    rgb[hi, 1] = 1.0 - t2[hi]     # green falls

    return rgb


# ---------------- Open3D helpers ----------------

def _write_colored_ply(path: str, points: np.ndarray, colors_0_1: np.ndarray) -> None:
    if o3d is None:
        raise RuntimeError("open3d is required to write PLY. Please `pip install open3d`.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_0_1, 0.0, 1.0).astype(np.float64))
    o3d.io.write_point_cloud(path, pcd)

def _show_pcd(title: str, points: np.ndarray, colors_0_1: np.ndarray) -> None:
    if o3d is None:
        raise RuntimeError("open3d is required to visualize. Please `pip install open3d`.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_0_1, 0.0, 1.0).astype(np.float64))
    o3d.visualization.draw_geometries([pcd], window_name=title)


# ---------------- Public API ----------------

def build_label_heatmaps(
    *,
    primitives_json_path: str,
    ply_path: str,
    cluster_ids_path: str,
    out_dir: str,
    voxel_size: Optional[float] = None,
    voxel_size_scale: float = 0.5,   # <1 => smaller voxels (more detailed heat)
    min_points_per_label: int = 200,
    show_windows: bool = True,
    max_labels_to_show: int = 12,
    show_combined: bool = True,
) -> Dict[str, Any]:
    """
    Build per-label heatmap PLYs and optionally visualize.

    voxel_size:
      - if None: auto estimated from shape diagonal / target_bins
      - then multiplied by voxel_size_scale (default 0.5 -> smaller voxels)

    Returns:
      dict with summary_json, voxel_size, labels_count
    """
    os.makedirs(out_dir, exist_ok=True)

    pts = _load_points_ply(ply_path)
    N = pts.shape[0]

    cluster_ids = _load_cluster_ids(cluster_ids_path, expected_n=N)
    c2l = _cluster_to_label_map(primitives_json_path)
    point_labels = np.array([c2l.get(int(cid), "unknown") for cid in cluster_ids], dtype=object)

    if voxel_size is None:
        voxel_size = _auto_voxel_size(pts, target_bins=160)
        voxel_size = float(voxel_size) * float(voxel_size_scale)
    voxel_size = float(voxel_size)

    inv, uniq_vox = _voxel_inv(pts, voxel_size=voxel_size)
    M = len(uniq_vox)

    # total points per voxel
    total_per_vox = np.bincount(inv, minlength=M).astype(np.float32)

    uniq_labels = sorted(set(map(str, point_labels.tolist())))
    label_infos = []
    payload_for_vis = []  # (label, colors, ply_path, stats)

    for lab in uniq_labels:
        idx = np.where(point_labels == lab)[0]
        if idx.size < min_points_per_label:
            continue

        inv_lab = inv[idx]
        lab_per_vox = np.bincount(inv_lab, minlength=M).astype(np.float32)

        heat_vox = np.zeros((M,), dtype=np.float32)
        ok = total_per_vox > 0
        heat_vox[ok] = lab_per_vox[ok] / total_per_vox[ok]

        heat_pt = heat_vox[inv]  # (N,)
        colors = _color_heat_red_green_black(heat_pt)

        lab_dir = os.path.join(out_dir, "heatmaps", _sanitize_name(lab))
        os.makedirs(lab_dir, exist_ok=True)
        out_ply = os.path.join(lab_dir, f"heat_map_{_sanitize_name(lab)}.ply")
        _write_colored_ply(out_ply, pts, colors)

        lab_info = {
            "label": lab,
            "label_point_count": int(idx.size),
            "voxel_size": float(voxel_size),
            "voxels_total": int(M),
            "voxels_with_label": int(np.sum(lab_per_vox > 0)),
            "voxels_full_label": int(np.sum((lab_per_vox > 0) & (np.isclose(lab_per_vox, total_per_vox)))),
            "heat_point_mean": float(np.mean(heat_pt)),
            "heat_point_p95": float(np.percentile(heat_pt, 95)),
            "heat_point_max": float(np.max(heat_pt)),
            "heat_ply": os.path.abspath(out_ply),
        }
        _save_json(os.path.join(lab_dir, "summary.json"), lab_info)

        label_infos.append(lab_info)
        payload_for_vis.append((lab, colors, out_ply, lab_info))

    # sort by label size
    label_infos = sorted(label_infos, key=lambda x: x["label_point_count"], reverse=True)
    payload_for_vis = sorted(payload_for_vis, key=lambda x: x[3]["label_point_count"], reverse=True)

    summary = {
        "inputs": {
            "primitives_json": os.path.abspath(primitives_json_path),
            "ply_path": os.path.abspath(ply_path),
            "cluster_ids_path": os.path.abspath(cluster_ids_path),
        },
        "voxel_size": float(voxel_size),
        "voxel_size_scale": float(voxel_size_scale),
        "min_points_per_label": int(min_points_per_label),
        "labels": label_infos,
        "note": "Per-label heatmap: heat(voxel)=#label_points/#total_points. Colors: 1->red, 0.5->green, 0->black.",
    }
    summary_path = os.path.join(out_dir, "heatmaps_summary.json")
    _save_json(summary_path, summary)

    print("\n[HEAT_MAP] Done.")
    print("[HEAT_MAP] voxel_size:", float(voxel_size), "(scale=", float(voxel_size_scale), ")")
    print("[HEAT_MAP] wrote:", summary_path)
    if len(label_infos) == 0:
        print("[HEAT_MAP] note: no labels passed min_points_per_label =", min_points_per_label)
    else:
        print("[HEAT_MAP] labels:", len(label_infos))
        print("[HEAT_MAP] top-5 labels by point_count:")
        for r in label_infos[:5]:
            print(f"  - {r['label']}: points={r['label_point_count']} heat_mean={r['heat_point_mean']:.3f}")

    # --- visualization ---
    if show_windows and o3d is not None and len(payload_for_vis) > 0:
        to_show = payload_for_vis[: int(max_labels_to_show)]
        for lab, colors, _, info in to_show:
            title = (f"HeatMap | {lab} | voxel={info['voxel_size']:.4g} "
                     f"| mean={info['heat_point_mean']:.3f} p95={info['heat_point_p95']:.3f}")
            print(f"[HEAT_MAP][VIS] opening: {title}")
            _show_pcd(title, pts, colors)

        if show_combined:
            # combined: max heat across shown labels (reconstruct heat from colors)
            heat_max = np.zeros((N,), dtype=np.float32)
            for _, colors, _, _ in to_show:
                r = colors[:, 0]
                g = colors[:, 1]
                heat = np.where(r > 1e-6, 0.5 + 0.5 * r, 0.5 * g)
                heat_max = np.maximum(heat_max, heat)

            col = _color_heat_red_green_black(heat_max)
            title = "HeatMap COMBINED (max over shown labels) | red=full, green=half, black=none"
            print(f"[HEAT_MAP][VIS] opening: {title}")
            _show_pcd(title, pts, col)

    return {
        "summary_json": summary_path,
        "voxel_size": float(voxel_size),
        "labels_count": len(label_infos),
    }
