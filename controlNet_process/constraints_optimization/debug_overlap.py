#!/usr/bin/env python3
"""
constraints_optimization/debug_overlap.py

Debug tool for bounding boxes:
1) Visualize ALL bounding boxes in ONE Open3D window (thick edges).
2) Print each bounding box WORLD AABB min/max (x,y,z).
3) Compute pairwise WORLD-AABB intersection volumes and report overlaps.

Usage:
  python -m constraints_optimization.debug_overlap \
    --bbox_json sketch/dsl_optimize/optimize_iteration/iter_000/heat_map/pca_bboxes/pca_bboxes.json \
    --radius 0.003 --topk 30
"""

import os
import json
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# OBB -> corners -> WORLD AABB
# -----------------------------------------------------------------------------

def _obb_corners_world(center: np.ndarray, R: np.ndarray, extent: np.ndarray) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    e = np.asarray(extent, dtype=np.float64).reshape(3)
    h = 0.5 * e

    signs = np.array(
        [[-1, -1, -1],
         [-1, -1,  1],
         [-1,  1, -1],
         [-1,  1,  1],
         [ 1, -1, -1],
         [ 1, -1,  1],
         [ 1,  1, -1],
         [ 1,  1,  1]],
        dtype=np.float64,
    )
    corners_local = signs * h[None, :]
    corners_world = (Rm @ corners_local.T).T + c[None, :]
    return corners_world

def _obb_world_aabb(center: np.ndarray, R: np.ndarray, extent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    corners = _obb_corners_world(center, R, extent)
    mn = corners.min(axis=0)
    mx = corners.max(axis=0)
    return mn, mx


# -----------------------------------------------------------------------------
# AABB overlap volume
# -----------------------------------------------------------------------------

def _pairwise_overlap_volume(mn1: np.ndarray, mx1: np.ndarray, mn2: np.ndarray, mx2: np.ndarray) -> float:
    omax = np.minimum(mx1, mx2)
    omin = np.maximum(mn1, mn2)
    oext = omax - omin
    oext = np.maximum(oext, 0.0)
    return float(oext[0] * oext[1] * oext[2])


# -----------------------------------------------------------------------------
# Thick edge rendering (cylinders per edge using LineSet connectivity)
# -----------------------------------------------------------------------------

def _cylinder_between(p0: np.ndarray, p1: np.ndarray, radius: float, color) -> "o3d.geometry.TriangleMesh":
    v = p1 - p0
    length = float(np.linalg.norm(v))
    if length < 1e-10:
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=float(radius), height=length)
    cyl.compute_vertex_normals()

    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v_hat = v / length

    axis = np.cross(z, v_hat)
    axis_norm = float(np.linalg.norm(axis))
    dot = float(np.clip(np.dot(z, v_hat), -1.0, 1.0))
    angle = float(np.arccos(dot))

    if axis_norm > 1e-10 and angle > 1e-10:
        axis = axis / axis_norm
        Rm = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cyl.rotate(Rm, center=(0.0, 0.0, 0.0))

    cyl.translate((p0 + p1) * 0.5)
    cyl.paint_uniform_color(color)
    return cyl

def _thick_obb_boundary_meshes(
    obb: "o3d.geometry.OrientedBoundingBox",
    *,
    radius: float,
    color=(0.0, 0.0, 1.0),
) -> List["o3d.geometry.TriangleMesh"]:
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    pts = np.asarray(ls.points, dtype=np.float64)
    lines = np.asarray(ls.lines, dtype=np.int64)

    meshes = []
    for a, b in lines:
        m = _cylinder_between(pts[a], pts[b], radius=radius, color=color)
        if m is not None:
            meshes.append(m)
    return meshes


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if o3d is None:
        raise RuntimeError("open3d required. pip install open3d")

    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox_json", required=True, help="Path to pca_bboxes.json (or optimized bbox json).")
    ap.add_argument("--radius", type=float, default=0.003, help="Edge thickness in world units.")
    ap.add_argument("--max_boxes", type=int, default=9999, help="Optional cap.")
    ap.add_argument("--topk", type=int, default=30, help="Show top-K overlaps.")
    args = ap.parse_args()

    bbox_json = args.bbox_json
    payload = _load_json(bbox_json)
    labels = payload.get("labels", [])
    if not labels:
        print("[DBG] No labels found in:", bbox_json)
        return

    labels = labels[: int(args.max_boxes)]

    print("\n[DBG] bbox_json:", os.path.abspath(bbox_json))
    print("[DBG] num_boxes:", len(labels))

    # ---- compute AABBs ----
    mins = []
    maxs = []
    names = []
    obbs = []

    print("\n[DBG] WORLD AABB mins/maxs per box:")
    for i, rec in enumerate(labels):
        name = rec.get("label", rec.get("sanitized", f"box_{i}"))
        obb_d = rec.get("obb", {})
        center = np.asarray(obb_d["center"], dtype=np.float64)
        Rm = np.asarray(obb_d["R"], dtype=np.float64)
        extent = np.asarray(obb_d["extent"], dtype=np.float64)

        mn, mx = _obb_world_aabb(center, Rm, extent)
        mins.append(mn)
        maxs.append(mx)
        names.append(name)

        obbs.append(o3d.geometry.OrientedBoundingBox(center=center, R=Rm, extent=extent))

        print(
            f"[BOX {i:02d}] {name}\n"
            f"  AABB min = ({mn[0]:.6g}, {mn[1]:.6g}, {mn[2]:.6g})\n"
            f"  AABB max = ({mx[0]:.6g}, {mx[1]:.6g}, {mx[2]:.6g})\n"
            f"  extent   = ({extent[0]:.6g}, {extent[1]:.6g}, {extent[2]:.6g})"
        )

    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)

    # ---- pairwise overlap volumes ----
    n = mins.shape[0]
    pairs = n * (n - 1) // 2

    overlaps = []  # (inter_vol, i, j)
    inter_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            v = _pairwise_overlap_volume(mins[i], maxs[i], mins[j], maxs[j])
            if v > 0.0:
                overlaps.append((float(v), int(i), int(j)))
                inter_sum += float(v)

    overlaps.sort(key=lambda t: t[0], reverse=True)

    print("\n[DBG] Pairwise WORLD-AABB overlap volumes:")
    print(f"[DBG] pairs_total={pairs}")
    print(f"[DBG] overlap_pairs(inter_vol>0)={len(overlaps)}")
    print(f"[DBG] inter_vol_sum={inter_sum:.9g}")
    if len(overlaps) == 0:
        print("[DBG] DO THEY OVERLAP AT ALL?  NO (no pair has positive AABB intersection volume)")
    else:
        print("[DBG] DO THEY OVERLAP AT ALL?  YES (at least one pair has positive AABB intersection volume)")

    topk = min(int(args.topk), len(overlaps))
    if topk > 0:
        print(f"\n[DBG] Top {topk} overlaps (by inter_vol):")
        for k in range(topk):
            v, i, j = overlaps[k]
            print(f"  inter_vol={v:.9g}  pair=({i},{j})  {names[i]}  <->  {names[j]}")

    # ---- visualization ----
    geoms: List[Any] = []
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    palette = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.5, 0.0),
        (0.5, 0.0, 1.0),
    ]

    for i, obb in enumerate(obbs):
        color = palette[i % len(palette)]
        geoms.extend(_thick_obb_boundary_meshes(obb, radius=float(args.radius), color=color))

    print("\n[DBG] Opening Open3D window with ALL boxes...")
    o3d.visualization.draw_geometries(geoms, window_name="DEBUG: All OBBs (thick edges)")


if __name__ == "__main__":
    main()
