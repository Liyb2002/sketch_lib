# constraints_extraction/via_pca_bbx.py
#!/usr/bin/env python3
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _normalize_primitives_json(raw: Any) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Accepts:
      - {"source_ply": "...", "primitives": [...]}
      - {"primitives": [...]}
      - [...]
    Returns: (source_ply, primitives_list)
    """
    if isinstance(raw, list):
        return None, raw
    if isinstance(raw, dict) and "primitives" in raw and isinstance(raw["primitives"], list):
        return raw.get("source_ply", None), raw["primitives"]
    raise ValueError(
        f"[Vis] Unsupported primitives JSON format. Got type={type(raw)}"
        + (f" with keys={list(raw.keys())}" if isinstance(raw, dict) else "")
    )


def _make_obb(record: Dict[str, Any]) -> o3d.geometry.OrientedBoundingBox:
    params = record.get("parameters", {})
    center = np.asarray(params.get("center", [0, 0, 0]), dtype=np.float64).reshape(3)
    extent = np.asarray(params.get("extent", [0, 0, 0]), dtype=np.float64).reshape(3)
    R = np.asarray(params.get("rotation", np.eye(3).tolist()), dtype=np.float64).reshape(3, 3)
    return o3d.geometry.OrientedBoundingBox(center, R, extent)


def run_visualization(
    primitives_json_path: str,
    ply_path: Optional[str] = None,
    cluster_ids_path: Optional[str] = None,
    *,
    background_keep_ratio: float = 1.0,
    show_all_boxes_faint: bool = False,
):
    """
    Visualize per-label:
      - The point cloud (gray background)
      - Points for this label (red)
      - OBBs for clusters belonging to this label (wireframe boxes)

    You MUST provide cluster_ids_path to highlight points for a label.
    `ply_path` can be omitted if primitives JSON contains "source_ply".
    """
    if not os.path.exists(primitives_json_path):
        raise FileNotFoundError(f"Missing primitives json: {primitives_json_path}")

    raw = _load_json(primitives_json_path)
    source_ply, primitives = _normalize_primitives_json(raw)

    if ply_path is None:
        ply_path = source_ply
    if ply_path is None or not os.path.exists(ply_path):
        raise FileNotFoundError(
            f"Missing ply_path. Provided={ply_path} ; source_ply(from json)={source_ply}"
        )

    if cluster_ids_path is None or not os.path.exists(cluster_ids_path):
        raise FileNotFoundError(
            "cluster_ids_path is required to overlay label-marked points.\n"
            f"Got: {cluster_ids_path}"
        )

    print(f"[Vis] Loading shape: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        print("[Vis] Empty point cloud.")
        return

    cluster_ids = np.load(cluster_ids_path).reshape(-1)
    if cluster_ids.shape[0] != points.shape[0]:
        raise RuntimeError(
            f"[Vis] cluster_ids length {cluster_ids.shape[0]} != points {points.shape[0]}.\n"
            "cluster_ids.npy must align point-for-point with the PLY."
        )

    # Build label -> list of primitive records
    label_to_recs: Dict[str, List[Dict[str, Any]]] = {}
    for r in primitives:
        label = str(r.get("label", "unknown"))
        label_to_recs.setdefault(label, []).append(r)

    labels = sorted(label_to_recs.keys())
    print(f"[Vis] Labels: {labels}")

    # Precompute all OBB line sets (optional faint display)
    all_box_lines: List[o3d.geometry.LineSet] = []
    if show_all_boxes_faint:
        for r in primitives:
            obb = _make_obb(r)
            ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            # faint blue-ish
            ls.paint_uniform_color([0.2, 0.2, 0.8])
            all_box_lines.append(ls)

    # Base background point cloud
    bg_pcd = o3d.geometry.PointCloud()
    bg_pcd.points = o3d.utility.Vector3dVector(points)

    # If user wants fewer background points for speed
    if background_keep_ratio < 1.0:
        background_keep_ratio = float(max(0.01, min(1.0, background_keep_ratio)))
        n = points.shape[0]
        keep = int(n * background_keep_ratio)
        idx = np.random.choice(n, size=keep, replace=False)
        bg_pcd = bg_pcd.select_by_index(idx)
        print(f"[Vis] Background downsample: keep_ratio={background_keep_ratio} -> {keep} points")

    bg_colors = np.zeros((np.asarray(bg_pcd.points).shape[0], 3), dtype=np.float64) + 0.6
    bg_pcd.colors = o3d.utility.Vector3dVector(bg_colors)

    for label in labels:
        recs = label_to_recs[label]

        # Collect cluster ids for this label from primitives
        cids = sorted({int(r.get("cluster_id", -1)) for r in recs if int(r.get("cluster_id", -1)) >= 0})
        if len(cids) == 0:
            continue

        # Select points for this label by cluster ids
        mask = np.isin(cluster_ids, np.array(cids, dtype=cluster_ids.dtype))
        sel_idx = np.where(mask)[0]
        if sel_idx.size == 0:
            print(f"[Vis][WARN] Label '{label}' has clusters {cids} but selects 0 points.")
            continue

        sel_pcd = pcd.select_by_index(sel_idx.tolist())
        sel_colors = np.zeros((np.asarray(sel_pcd.points).shape[0], 3), dtype=np.float64)
        # red
        sel_colors[:, 0] = 1.0
        sel_pcd.colors = o3d.utility.Vector3dVector(sel_colors)

        # Build OBB line sets for this label
        label_boxes: List[o3d.geometry.LineSet] = []
        for r in recs:
            obb = _make_obb(r)
            ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            # green boxes for current label
            ls.paint_uniform_color([0.0, 1.0, 0.0])
            label_boxes.append(ls)

        print(
            f"\n[Vis] Label: {label} | clusters={len(cids)} | selected_points={sel_idx.size} | boxes={len(label_boxes)}"
        )
        print("[Vis] Close the window to continue to the next label...")

        geoms: List[o3d.geometry.Geometry] = [bg_pcd, sel_pcd]
        if show_all_boxes_faint:
            geoms.extend(all_box_lines)
        geoms.extend(label_boxes)

        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"PCA BBX by Label: {label}",
            width=1280,
            height=720,
        )
