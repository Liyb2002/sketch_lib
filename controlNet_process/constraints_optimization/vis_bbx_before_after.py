#!/usr/bin/env python3
import json
import os
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import open3d as o3d


def _load_primitives(primitives_json_path: str) -> List[Dict[str, Any]]:
    with open(primitives_json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "primitives" in data:
        return data["primitives"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected primitives JSON format in {primitives_json_path}")


def _prim_map_by_cluster_id(prims: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for p in prims:
        try:
            cid = int(p.get("cluster_id", -1))
        except Exception:
            continue
        out[cid] = p
    return out


def _obb_from_params(params: Dict[str, Any]) -> o3d.geometry.OrientedBoundingBox:
    center = np.array(params.get("center", [0, 0, 0]), dtype=np.float64)
    extent = np.array(params.get("extent", [0, 0, 0]), dtype=np.float64)
    R = np.array(params.get("rotation", np.eye(3)), dtype=np.float64)
    return o3d.geometry.OrientedBoundingBox(center, R, extent)


def _colorize_points_by_cluster_id(
    pcd: o3d.geometry.PointCloud,
    cluster_ids: np.ndarray,
    target_cluster_id: int,
    *,
    active_rgb=(1.0, 0.6, 0.1),
    inactive_rgb=(0.05, 0.05, 0.05),
) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    colors = np.zeros((pts.shape[0], 3), dtype=np.float64)
    colors[:] = np.array(inactive_rgb, dtype=np.float64)
    mask = (cluster_ids == target_cluster_id)
    colors[mask] = np.array(active_rgb, dtype=np.float64)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    out.colors = o3d.utility.Vector3dVector(colors)
    return out


def _load_points_and_cluster_ids(ply_path: str, cluster_ids_path: str) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Missing ply: {ply_path}")
    if not os.path.exists(cluster_ids_path):
        raise FileNotFoundError(f"Missing cluster_ids npy: {cluster_ids_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    cluster_ids = np.load(cluster_ids_path).reshape(-1)
    pts = np.asarray(pcd.points)
    if pts.shape[0] != cluster_ids.shape[0]:
        raise RuntimeError("cluster_ids.npy must align 1:1 with the PLY points.")
    return pcd, cluster_ids


def run_before_after_vis_per_label(
    before_primitives_json: str,
    after_primitives_json: str,
    ply_path: str,
    cluster_ids_path: str,
):
    before = _load_primitives(before_primitives_json)
    after = _load_primitives(after_primitives_json)

    before_map = _prim_map_by_cluster_id(before)
    after_map = _prim_map_by_cluster_id(after)

    pcd, cluster_ids = _load_points_and_cluster_ids(ply_path, cluster_ids_path)

    # Iterate in stable order by cluster id present in BOTH
    common_cids = sorted(set(before_map.keys()) & set(after_map.keys()))
    print(f"[VIS] Per-label views: {len(common_cids)} clusters")

    for cid in common_cids:
        b = before_map[cid]
        a = after_map[cid]
        label = str(a.get("label", b.get("label", "unknown")))

        print("\n" + "=" * 80)
        print(f"[VIS] label={label}  cluster_id={cid}")

        bp = b.get("parameters", {})
        ap = a.get("parameters", {})

        print("[BEFORE] center:", bp.get("center"), " extent:", bp.get("extent"))
        print("[AFTER ] center:", ap.get("center"), " extent:", ap.get("extent"))

        pcd_col = _colorize_points_by_cluster_id(pcd, cluster_ids, cid)

        obb_before = _obb_from_params(bp)
        obb_after = _obb_from_params(ap)

        # colors: before=blue-ish, after=orange-ish
        obb_before.color = (0.2, 0.5, 1.0)
        obb_after.color = (1.0, 0.6, 0.1)

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        o3d.visualization.draw_geometries(
            [pcd_col, obb_before, obb_after, coord],
            window_name=f"Per-label before/after: {label} (cid={cid})",
        )


def run_global_vis_all_boxes(
    primitives_json: str,
    ply_path: str,
    *,
    cluster_ids_path: Optional[str] = None,
    show_points: bool = True,
    points_rgb=(0.08, 0.08, 0.08),
):
    prims = _load_primitives(primitives_json)
    geoms = []

    if show_points:
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points)
        colors = np.zeros((pts.shape[0], 3), dtype=np.float64)
        colors[:] = np.array(points_rgb, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geoms.append(pcd)

    # draw all boxes
    for p in prims:
        params = p.get("parameters", {})
        obb = _obb_from_params(params)
        # deterministic-ish color by label hash
        label = str(p.get("label", "unknown"))
        h = abs(hash(label)) % 360
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.75, 1.0)
        obb.color = (r, g, b)
        geoms.append(obb)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geoms.append(coord)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Global view: all boxes together",
    )