#!/usr/bin/env python3
# graph_building/vis_bbx.py

import numpy as np
import open3d as o3d


def _as_np3(x):
    return np.asarray(x, dtype=np.float64).reshape(3,)


def _as_np33(x):
    return np.asarray(x, dtype=np.float64).reshape(3, 3)


def _compute_obb_corners(center, axes, extents):
    c = center.reshape(3,)
    e = extents.reshape(3,)
    a0, a1, a2 = axes[:, 0], axes[:, 1], axes[:, 2]

    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                p = c + sx * e[0] * a0 + sy * e[1] * a1 + sz * e[2] * a2
                corners.append(p)
    return np.stack(corners, axis=0)


def _make_obb_lineset(center, axes, extents, color=(1.0, 0.0, 0.0)):
    center = _as_np3(center)
    axes = _as_np33(axes)
    extents = _as_np3(extents)

    corners = _compute_obb_corners(center, axes, extents)

    lines = np.array([
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [3, 7],
        [4, 5], [4, 6],
        [5, 7],
        [6, 7],
    ], dtype=np.int32)

    cols = np.tile(np.asarray(color, dtype=np.float64)[None, :], (lines.shape[0], 1))

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls


def vis_bboxes_by_label(
    pts,
    assigned_ids,
    bboxes_by_name,
    colors=None,
    window_prefix="BBX",
    ignore_unknown=False,
    show_all_bboxes=True,
    fade_all_alpha=0.5,
):
    pts = np.asarray(pts, dtype=np.float64)
    assigned_ids = np.asarray(assigned_ids, dtype=np.int32).reshape(-1)

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)

    keys = sorted(list(bboxes_by_name.keys()))
    if ignore_unknown:
        keys = [k for k in keys if not str(k).startswith("unknown_")]

    # ---------- All bboxes together ----------
    if show_all_bboxes:

        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(pts)

        if colors is not None:
            faded = colors * fade_all_alpha
        else:
            faded = np.ones((pts.shape[0], 3)) * fade_all_alpha

        pcd_all.colors = o3d.utility.Vector3dVector(faded)

        geoms = [pcd_all]

        for name in keys:
            obb = bboxes_by_name[name]["obb_pca"]
            geoms.append(
                _make_obb_lineset(
                    obb["center"], obb["axes"], obb["extents"],
                    color=(1.0, 0.0, 0.0)  # red
                )
            )

        print(f"[VIS_BBX] All bboxes ({len(keys)})")
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"{window_prefix}: ALL_BBOXES",
            width=1400,
            height=900,
        )

    # ---------- Per-label ----------
    for name in keys:
        entry = bboxes_by_name[name]
        lid = int(entry["label_id"])
        mask = (assigned_ids == lid)

        if not np.any(mask):
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        base_cols = np.zeros((pts.shape[0], 3), dtype=np.float64)
        if colors is not None:
            base_cols[mask] = colors[mask]
        else:
            base_cols[mask] = np.array([0.8, 0.8, 0.8])

        pcd.colors = o3d.utility.Vector3dVector(base_cols)

        obb = entry["obb_pca"]
        ls = _make_obb_lineset(
            obb["center"], obb["axes"], obb["extents"],
            color=(1.0, 0.0, 0.0)  # red
        )

        print(f"[VIS_BBX] {name}  lid={lid}  n={int(mask.sum())}")

        o3d.visualization.draw_geometries(
            [pcd, ls],
            window_name=f"{window_prefix}: {name}",
            width=1400,
            height=900,
        )
