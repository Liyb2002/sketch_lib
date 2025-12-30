# AEP/vis.py
"""
AEP/vis.py

Visualize AEP neighbor changes with Open3D.

Shows three windows:
1) BEFORE:
   - fused_model.ply faint gray
   - target bbox red
   - all neighbor bboxes blue (light-ish)

2) AFTER:
   - fused_model.ply faint gray
   - target bbox red
   - unchanged neighbors: faint blue
   - changed neighbors: dark blue

3) DELTAS (focused):
   - fused_model.ply faint gray
   - target bbox red (after)
   - ONLY changed neighbors are shown, each with:
       - neighbor BEFORE bbox in faint blue
       - neighbor AFTER  bbox in dark blue
   This view is for quickly seeing the actual neighbor motion.

Notes:
- Open3D LineSet doesn't do transparency reliably across platforms.
  "Faint" is implemented as lighter RGB colors.
"""

from typing import Dict, List, Set, Any, Tuple
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


def _np3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)


def _make_aabb_lineset(mn: np.ndarray, mx: np.ndarray, color_rgb: Tuple[float, float, float]) -> "o3d.geometry.LineSet":
    mn = _np3(mn)
    mx = _np3(mx)

    corners = np.array([
        [mn[0], mn[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]],
        [mn[0], mx[1], mx[2]],
    ], dtype=np.float64)

    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    colors = np.tile(np.array(color_rgb, dtype=np.float64)[None, :], (lines.shape[0], 1))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _load_fused_ply_faint(path: str) -> "o3d.geometry.Geometry":
    # Prefer point cloud
    pcd = o3d.io.read_point_cloud(path)
    if pcd is not None and len(np.asarray(pcd.points)) > 0:
        pcd.paint_uniform_color([0.78, 0.78, 0.78])
        return pcd

    # Fallback triangle mesh
    m = o3d.io.read_triangle_mesh(path)
    if m is not None and len(np.asarray(m.vertices)) > 0:
        m.compute_vertex_normals()
        m.paint_uniform_color([0.78, 0.78, 0.78])
        return m

    return pcd


def _draw_window(title: str, geoms: List["o3d.geometry.Geometry"]) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=900)
    opt = vis.get_render_option()
    if opt is not None:
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        opt.point_size = 2.0
        opt.line_width = 5.0  # stronger bboxes
    for g in geoms:
        if g is not None:
            vis.add_geometry(g)
    vis.run()
    vis.destroy_window()


def show_aep_before_after(
    fused_ply_path: str,
    target_label: str,
    before_aabbs: Dict[str, Dict[str, List[float]]],
    after_aabbs: Dict[str, Dict[str, List[float]]],
    changed_neighbors: Set[str],
) -> None:
    if o3d is None:
        raise RuntimeError("open3d is required for visualization: pip install open3d")

    # Colors
    RED = (1.0, 0.1, 0.1)
    BLUE_LIGHT = (0.55, 0.70, 1.00)   # before neighbors
    BLUE_FAINT = (0.78, 0.86, 1.00)   # after unchanged neighbors
    BLUE_DARK = (0.05, 0.20, 0.85)    # after changed neighbors

    # ---------- 1) BEFORE ----------
    fused_before = _load_fused_ply_faint(fused_ply_path)
    geoms_before: List["o3d.geometry.Geometry"] = [fused_before]

    for lab, aabb in before_aabbs.items():
        mn = _np3(aabb["min"])
        mx = _np3(aabb["max"])
        if lab == target_label:
            geoms_before.append(_make_aabb_lineset(mn, mx, RED))
        else:
            geoms_before.append(_make_aabb_lineset(mn, mx, BLUE_LIGHT))

    # ---------- 2) AFTER ----------
    fused_after = _load_fused_ply_faint(fused_ply_path)
    geoms_after: List["o3d.geometry.Geometry"] = [fused_after]

    for lab, aabb in after_aabbs.items():
        mn = _np3(aabb["min"])
        mx = _np3(aabb["max"])
        if lab == target_label:
            geoms_after.append(_make_aabb_lineset(mn, mx, RED))
        else:
            if lab in changed_neighbors:
                geoms_after.append(_make_aabb_lineset(mn, mx, BLUE_DARK))
            else:
                geoms_after.append(_make_aabb_lineset(mn, mx, BLUE_FAINT))

    # ---------- 3) DELTAS (focused) ----------
    fused_deltas = _load_fused_ply_faint(fused_ply_path)
    geoms_deltas: List["o3d.geometry.Geometry"] = [fused_deltas]

    # show only "source edit" bbox (after)
    if target_label in after_aabbs:
        ta = after_aabbs[target_label]
        geoms_deltas.append(_make_aabb_lineset(_np3(ta["min"]), _np3(ta["max"]), RED))

    # for each changed neighbor, show before+after
    for lab in sorted(changed_neighbors):
        if lab not in before_aabbs or lab not in after_aabbs:
            continue
        b0 = before_aabbs[lab]
        b1 = after_aabbs[lab]
        geoms_deltas.append(_make_aabb_lineset(_np3(b0["min"]), _np3(b0["max"]), BLUE_FAINT))  # before
        geoms_deltas.append(_make_aabb_lineset(_np3(b1["min"]), _np3(b1["max"]), BLUE_DARK))   # after

    _draw_window("AEP BEFORE (target=red)", geoms_before)
    _draw_window("AEP AFTER (target=red, changed=dark blue, unchanged=faint blue)", geoms_after)
    _draw_window("AEP DELTAS (target=red; changed neighbors: before=faint blue, after=dark blue)", geoms_deltas)
