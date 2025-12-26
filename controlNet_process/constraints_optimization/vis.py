#!/usr/bin/env python3
"""
constraints_optimization/vis.py

Visualization:
- Load heatmap PLY (colored full-shape point cloud)
- Overlay PCA OBB as THICK BLUE BOUNDARY ONLY
  (cylinders per edge, using Open3D LineSet connectivity to avoid wrong corner ordering)

Also prints a quick value check:
- For each label heatmap, split its AABB into grid_res^3 bins (default 3x3x3)
- Print SUM of heat values per bin (and point counts)

No solid boxes.
"""

import json
from typing import Any, Dict, List

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _load_colored_ply(path: str) -> "o3d.geometry.PointCloud":
    if o3d is None:
        raise RuntimeError("open3d required")
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pcd


# ---------------------------------------------------------------------
# OBB restore
# ---------------------------------------------------------------------

def _obb_from_dict(d: Dict[str, Any]) -> "o3d.geometry.OrientedBoundingBox":
    if o3d is None:
        raise RuntimeError("open3d required")
    return o3d.geometry.OrientedBoundingBox(
        center=np.asarray(d["center"], dtype=np.float64),
        R=np.asarray(d["R"], dtype=np.float64),
        extent=np.asarray(d["extent"], dtype=np.float64),
    )


# ---------------------------------------------------------------------
# Heat decode (must match heat_map.py colormap exactly)
# ---------------------------------------------------------------------

def _heat_from_red_green_black(colors_0_1: np.ndarray) -> np.ndarray:
    """
    Reverse the piecewise colormap used in heat_map.py:

      if h <= 0.5:
        rgb = (0, 2h, 0)        -> g = 2h -> h = 0.5*g
      if h >= 0.5:
        rgb = (2h-1, 2-2h, 0)   -> r = 2h-1 -> h = 0.5 + 0.5*r

    Use r>0 as a stable branch selector (matches how colors are written).
    """
    c = np.asarray(colors_0_1, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"colors must be (N,3), got {c.shape}")
    r = c[:, 0]
    g = c[:, 1]
    h = np.where(r > 1e-6, 0.5 + 0.5 * r, 0.5 * g)
    return np.clip(h, 0.0, 1.0)


# ---------------------------------------------------------------------
# Value function: SUM of heat in 3x3x3 bins
# ---------------------------------------------------------------------

def compute_values(
    pcd: "o3d.geometry.PointCloud",
    *,
    grid_res: int = 3,
    label: str = "",
) -> None:
    """
    Split the heatmap point cloud AABB into grid_res^3 bins.
    Print:
      - total heat sum over all points
      - per-cell SUM of heat (NOT average) + point count

    This is the simplest “sum of values in this area” check.
    """
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float32)

    if pts.size == 0:
        print(f"[VALUES] label={label}: empty pcd")
        return

    heat = _heat_from_red_green_black(cols)  # (N,) in [0,1]
    total_sum = float(np.sum(heat))
    total_pts = int(pts.shape[0])

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)

    R = int(grid_res)
    extent = mx - mn
    # If an axis is degenerate, make its step nonzero so indexing works
    extent = np.where(np.abs(extent) < 1e-12, 1.0, extent)
    step = extent / float(R)

    # Bin index for each point
    idx = np.floor((pts - mn) / step).astype(np.int64)  # (N,3)
    idx = np.clip(idx, 0, R - 1)

    grid_sum = np.zeros((R, R, R), dtype=np.float64)
    grid_cnt = np.zeros((R, R, R), dtype=np.int64)

    for (i, j, k), h in zip(idx, heat):
        grid_sum[i, j, k] += float(h)
        grid_cnt[i, j, k] += 1

    print(f"\n[VALUES] label={label}  points={total_pts}  total_heat_sum={total_sum:.6f}")
    print(f"[VALUES] grid={R}x{R}x{R}  cell_sum(cnt)")
    for i in range(R):
        for j in range(R):
            row = []
            for k in range(R):
                row.append(f"{grid_sum[i,j,k]:.4f}({grid_cnt[i,j,k]})")
            print(f"  cell[{i},{j},:]: " + "  ".join(row))


# ---------------------------------------------------------------------
# Thick boundary rendering (cylinders per *correct* LineSet edge)
# ---------------------------------------------------------------------

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


def _thick_obb_boundary_from_lineset(
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


# ---------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------

def visualize_heatmaps_with_bboxes(
    *,
    heat_dir: str,                 # kept for compatibility with your launcher; not used internally
    bbox_json: str,
    max_labels_to_show: int = 12,
    darken_heatmap: float = 0.7,
    bbox_radius: float = 0.003,
    print_grid_values: bool = True,
    grid_res: int = 3,
) -> None:
    if o3d is None:
        raise RuntimeError("open3d required")

    payload = _load_json(bbox_json)
    labels = payload.get("labels", [])
    if not labels:
        print("[VIS] No labels to visualize.")
        return

    labels = sorted(labels, key=lambda r: int(r.get("points_used", 0)), reverse=True)
    labels = labels[: int(max_labels_to_show)]

    for rec in labels:
        label = rec.get("label", rec.get("sanitized", "unknown"))
        pcd = _load_colored_ply(rec["heat_ply"])

        # IMPORTANT: compute values BEFORE any darkening
        if bool(print_grid_values):
            compute_values(pcd, grid_res=int(grid_res), label=str(label))

        # Darken only for visualization
        if float(darken_heatmap) < 0.999:
            cols = np.asarray(pcd.colors)
            cols = np.clip(cols * float(darken_heatmap), 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(cols)

        obb = _obb_from_dict(rec["obb"])
        bbox_meshes = _thick_obb_boundary_from_lineset(
            obb,
            radius=float(bbox_radius),
            color=(0.0, 0.0, 1.0),
        )

        title = f"HeatMap + PCA BBox (BLUE) | {label} | used={rec.get('points_used')} | min_heat={rec.get('min_heat')}"
        print("[VIS] opening:", title)
        o3d.visualization.draw_geometries([pcd] + bbox_meshes, window_name=title)





def visualize_heatmaps_with_bboxes_before_after(
    *,
    heat_dir: str,                   # kept for compatibility; not used
    bbox_json_before: str,
    bbox_json_after: str,
    max_labels_to_show: int = 12,
    darken_heatmap: float = 0.7,
    bbox_radius: float = 0.003,
) -> None:
    """
    For each label (matched by label string):
      - show heatmap point cloud
      - overlay BEFORE bbox boundary (BLUE)
      - overlay AFTER  bbox boundary (MAGENTA)

    Useful to visually verify shrink-only optimization.
    """
    if o3d is None:
        raise RuntimeError("open3d required")

    before = _load_json(bbox_json_before)
    after  = _load_json(bbox_json_after)

    labels_b = before.get("labels", [])
    labels_a = after.get("labels", [])
    if not labels_b or not labels_a:
        print("[VIS_BA] Missing labels in before/after json.")
        return

    # Map by label string
    def _lab_key(rec: Dict[str, Any]) -> str:
        return str(rec.get("label", rec.get("sanitized", "unknown")))

    map_b = {_lab_key(r): r for r in labels_b}
    map_a = {_lab_key(r): r for r in labels_a}

    # Sort by BEFORE points_used (or fallback 0)
    keys = list(map_b.keys())
    keys = sorted(keys, key=lambda k: int(map_b[k].get("points_used", 0)), reverse=True)
    keys = keys[: int(max_labels_to_show)]

    for k in keys:
        if k not in map_a:
            print(f"[VIS_BA] skip '{k}' (not found in after json)")
            continue

        rec_b = map_b[k]
        rec_a = map_a[k]

        # Load heatmap PLY (use BEFORE's heat_ply path)
        pcd = _load_colored_ply(rec_b["heat_ply"])

        # darken heatmap for contrast
        if float(darken_heatmap) < 0.999:
            cols = np.asarray(pcd.colors)
            cols = np.clip(cols * float(darken_heatmap), 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(cols)

        obb_b = _obb_from_dict(rec_b["obb"])
        obb_a = _obb_from_dict(rec_a["obb"])

        # BEFORE = blue
        meshes_b = _thick_obb_boundary_from_lineset(
            obb_b,
            radius=float(bbox_radius),
            color=(0.0, 0.0, 1.0),
        )

        # AFTER = magenta
        meshes_a = _thick_obb_boundary_from_lineset(
            obb_a,
            radius=float(bbox_radius),
            color=(1.0, 0.0, 1.0),
        )

        used_b = rec_b.get("points_used", None)
        used_a = rec_a.get("points_used", None)

        title = f"BEFORE(BLUE) vs AFTER(MAGENTA) | {k} | used_b={used_b} used_a={used_a}"
        print("[VIS_BA] opening:", title)

        geoms = [pcd] + meshes_b + meshes_a
        o3d.visualization.draw_geometries(geoms, window_name=title)
