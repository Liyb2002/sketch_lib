#!/usr/bin/env python3
"""
constraints_optimization/pca_analysis.py

Load saved per-label heatmap PLYs (colored points), recover per-point heat,
and compute PCA-based oriented bounding boxes per label.

Output JSON schema (per label):
{
  "label": "Front Wheel",
  "sanitized": "front_wheel",
  "heat_ply": ".../heat_map_front_wheel.ply",
  "min_heat": 0.5,
  "points_used": 12345,
  "obb": {
    "center": [x,y,z],
    "R": [[...],[...],[...]],
    "extent": [ex,ey,ez]
  },
  "aabb": {
    "min_bound": [x,y,z],
    "max_bound": [x,y,z]
  }
}
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


# ---------------- heatmap PLY loading ----------------

def _load_colored_ply(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points).astype(np.float32)
    cols = np.asarray(pcd.colors).astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Bad points shape in {path}: {pts.shape}")
    if cols.shape != pts.shape:
        # Some writers may omit colors; handle gracefully
        if cols.size == 0:
            cols = np.zeros_like(pts, dtype=np.float32)
        else:
            raise ValueError(f"Bad colors shape in {path}: {cols.shape} vs points {pts.shape}")
    return pts, cols


def _colors_to_heat(colors_0_1: np.ndarray) -> np.ndarray:
    """
    Reverse the piecewise colormap used in your heat_map.py:

      h in [0,0.5]  -> (r=0, g=2h, b=0)
      h in [0.5,1]  -> (r=2(h-0.5), g=2(1-h), b=0)

    Recover approximate h from r/g.
    """
    c = np.clip(colors_0_1.astype(np.float32), 0.0, 1.0)
    r = c[:, 0]
    g = c[:, 1]

    # If red > 0 => h in [0.5,1], h = 0.5 + 0.5*r
    # Else => h in [0,0.5], h = 0.5*g
    h = np.where(r > 1e-6, 0.5 + 0.5 * r, 0.5 * g)
    return np.clip(h, 0.0, 1.0).astype(np.float32)


# ---------------- bbox compute ----------------

def _compute_obb(points: np.ndarray) -> "o3d.geometry.OrientedBoundingBox":
    """
    PCA-ish OBB (Open3D uses PCA internally for create_from_points).
    """
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")
    p = o3d.utility.Vector3dVector(points.astype(np.float64))
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(p)
    return obb


def _obb_to_dict(obb: "o3d.geometry.OrientedBoundingBox") -> Dict[str, Any]:
    center = np.asarray(obb.center).astype(float).tolist()
    R = np.asarray(obb.R).astype(float).tolist()
    extent = np.asarray(obb.extent).astype(float).tolist()
    return {"center": center, "R": R, "extent": extent}


def _aabb_to_dict(points: np.ndarray) -> Dict[str, Any]:
    mn = points.min(axis=0).astype(float).tolist()
    mx = points.max(axis=0).astype(float).tolist()
    return {"min_bound": mn, "max_bound": mx}


# ---------------- public API ----------------

def compute_pca_bounding_boxes(
    *,
    heat_dir: str,
    out_json: str,
    min_heat: float = 0.5,
    min_points: int = 200,
    max_labels: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Finds heatmap PLYs under:
      heat_dir/heatmaps/<label>/heat_map_<label>.ply

    For each label PLY, recover heat from colors and compute OBB from points
    with heat >= min_heat.

    Writes out_json and returns the dict.
    """
    if o3d is None:
        raise RuntimeError("open3d is required. Please `pip install open3d`.")

    heatmaps_root = os.path.join(heat_dir, "heatmaps")
    if not os.path.isdir(heatmaps_root):
        raise FileNotFoundError(f"Missing heatmaps folder: {heatmaps_root}")

    # Collect all heat map ply files
    ply_entries: List[Tuple[str, str]] = []  # (sanitized_label, ply_path)
    for sub in sorted(os.listdir(heatmaps_root)):
        subdir = os.path.join(heatmaps_root, sub)
        if not os.path.isdir(subdir):
            continue

        # prefer any file that starts with heat_map_ and ends with .ply
        candidates = [f for f in os.listdir(subdir) if f.startswith("heat_map_") and f.endswith(".ply")]
        if not candidates:
            continue
        candidates.sort()
        ply_entries.append((sub, os.path.join(subdir, candidates[0])))

    if not ply_entries:
        raise FileNotFoundError(f"No heat_map_*.ply found under: {heatmaps_root}")

    # Load summary to recover original label names (optional)
    summary_path = os.path.join(heat_dir, "heatmaps_summary.json")
    summary = None
    if os.path.exists(summary_path):
        try:
            summary = _load_json(summary_path)
        except Exception:
            summary = None

    # Map sanitized -> original label if present in summary
    sanitized_to_label: Dict[str, str] = {}
    if isinstance(summary, dict) and isinstance(summary.get("labels", None), list):
        for item in summary["labels"]:
            lab = str(item.get("label", "unknown"))
            sanitized_to_label[_sanitize_name(lab)] = lab

    results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    # Optional cap
    if max_labels is not None:
        ply_entries = ply_entries[: int(max_labels)]

    for sanitized, ply_path in ply_entries:
        pts, cols = _load_colored_ply(ply_path)
        heat = _colors_to_heat(cols)

        keep = heat >= float(min_heat)
        used = pts[keep]

        label = sanitized_to_label.get(sanitized, sanitized)

        if used.shape[0] < int(min_points):
            skipped.append({
                "label": label,
                "sanitized": sanitized,
                "heat_ply": os.path.abspath(ply_path),
                "reason": f"too_few_points_above_min_heat ({used.shape[0]} < {min_points})",
                "min_heat": float(min_heat),
                "points_used": int(used.shape[0]),
            })
            continue

        obb = _compute_obb(used)

        rec = {
            "label": label,
            "sanitized": sanitized,
            "heat_ply": os.path.abspath(ply_path),
            "min_heat": float(min_heat),
            "points_used": int(used.shape[0]),
            "obb": _obb_to_dict(obb),
            "aabb": _aabb_to_dict(used),
        }
        results.append(rec)

    payload = {
        "heat_dir": os.path.abspath(heat_dir),
        "out_json": os.path.abspath(out_json),
        "min_heat": float(min_heat),
        "min_points": int(min_points),
        "labels_count": int(len(results)),
        "labels_skipped": skipped,
        "labels": results,
        "note": "OBB is computed from points with heat>=min_heat (heat recovered from PLY colors).",
    }

    _save_json(out_json, payload)

    print("\n[PCA_BBOX] Done.")
    print("[PCA_BBOX] wrote:", out_json)
    print("[PCA_BBOX] labels:", len(results), "skipped:", len(skipped))
    if skipped:
        print("[PCA_BBOX] first skipped:", skipped[0].get("label"), "-", skipped[0].get("reason"))

    return payload
