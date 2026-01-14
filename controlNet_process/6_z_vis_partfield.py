#!/usr/bin/env python3
"""
vis_partseg_open3d_overlay.py

Visualize a segmented point cloud PLY (RGB-colored clusters) in Open3D,
overlaying ONE cluster at a time on top of the full shape.

For each cluster id:
- Show ALL points in black (the whole shape context)
- Show the selected cluster in its original color (or a bright color if you want)
- Opens an Open3D window; close it to advance to next cluster
- Runs K times (default 20)
- Does NOT save anything

Usage:
  python vis_partseg_open3d_overlay.py
  python vis_partseg_open3d_overlay.py --ply sketch/3d_reconstruction/clustering_k20_points.ply --k 20
"""

import os
import argparse
import numpy as np
from plyfile import PlyData
import open3d as o3d


def load_ply_xyz_rgb(ply_path: str):
    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ValueError(f"No 'vertex' element found in {ply_path}")
    v = ply["vertex"]
    names = v.data.dtype.names

    for k in ("x", "y", "z"):
        if k not in names:
            raise ValueError(f"PLY vertex missing '{k}'. Found fields: {names}")

    for k in ("red", "green", "blue"):
        if k not in names:
            raise ValueError(
                f"PLY vertex missing '{k}'. This viewer expects a segmented/colorized PLY.\n"
                f"Found fields: {names}"
            )

    pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb255 = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    return pts, rgb255


def labels_from_rgb255(rgb255: np.ndarray, k: int):
    """
    Primary: recover integer cluster ids from the deterministic palette used by labels_to_colors():

      r = (label * 97 + 17) % 256
      g = (label * 57 + 101) % 256
      b = (label * 193 + 43) % 256

    Fallback: if colors don't match that palette (e.g. hash-based colors), assign labels by unique RGB tuples.
    """
    # --- primary palette decoding ---
    lut = {}
    for lbl in range(k):
        r = (lbl * 97 + 17) % 256
        g = (lbl * 57 + 101) % 256
        b = (lbl * 193 + 43) % 256
        lut[(r, g, b)] = lbl

    labels = np.full((rgb255.shape[0],), -1, dtype=np.int32)
    keys = [tuple(c) for c in rgb255.tolist()]
    for i, key in enumerate(keys):
        labels[i] = lut.get(key, -1)

    unknown = int(np.sum(labels < 0))
    if unknown == 0:
        return labels

    # --- fallback: unique-color clustering ---
    uniq, inv = np.unique(rgb255.reshape(-1, 3), axis=0, return_inverse=True)

    # Prefer to treat pure black as "unknown"/background if present
    is_black = np.all(uniq == np.array([0, 0, 0], dtype=np.uint8), axis=1)
    uniq_nonblack = uniq[~is_black]

    if uniq_nonblack.shape[0] > k:
        # Keep the k most frequent colors (excluding black)
        counts = np.bincount(inv)
        nonblack_indices = np.where(~is_black)[0]
        nonblack_counts = counts[nonblack_indices]
        topk_idx = nonblack_indices[np.argsort(-nonblack_counts)[:k]]
        keep_mask = np.zeros((uniq.shape[0],), dtype=bool)
        keep_mask[topk_idx] = True
        # black (if exists) remains not kept -> label -1
    else:
        keep_mask = ~is_black

    # Build mapping from color -> compact label [0..k-1]
    kept_colors = uniq[keep_mask]
    # Stable ordering for reproducibility
    kept_colors = kept_colors[np.lexsort((kept_colors[:, 2], kept_colors[:, 1], kept_colors[:, 0]))]

    color_to_lbl = {tuple(c.tolist()): i for i, c in enumerate(kept_colors)}
    labels_fb = np.full((rgb255.shape[0],), -1, dtype=np.int32)
    for i, key in enumerate(keys):
        labels_fb[i] = color_to_lbl.get(key, -1)

    fb_unknown = int(np.sum(labels_fb < 0))
    print(
        f"[WARN] Palette decode did not match ({unknown} unknown). Using unique-color fallback.\n"
        f"       Unique non-black colors found: {uniq_nonblack.shape[0]}, using up to k={k}.\n"
        f"       Fallback unknown (e.g. black/background or extra colors): {fb_unknown}."
    )
    return labels_fb


def make_pcd(points: np.ndarray, colors01: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors01.astype(np.float64))
    return pcd


def show_cluster_overlay(points, rgb255, labels, cluster_id: int):
    # Background: whole shape in black
    black = np.zeros((points.shape[0], 3), dtype=np.float32)
    pcd_all = make_pcd(points, black)

    # Foreground: selected cluster in its original color (from the PLY)
    mask = labels == cluster_id
    if not np.any(mask):
        print(f"[WARN] cluster {cluster_id}: no points (skipping)")
        o3d.visualization.draw_geometries([pcd_all], window_name=f"Cluster {cluster_id} (empty) — close to continue")
        return

    pts_c = points[mask]
    col_c = rgb255[mask].astype(np.float32) / 255.0
    pcd_cluster = make_pcd(pts_c, col_c)

    title = f"Cluster {cluster_id} overlay — close window to continue"
    o3d.visualization.draw_geometries([pcd_all, pcd_cluster], window_name=title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ply",
        type=str,
        default=os.path.join("sketch", "3d_reconstruction", "clustering_k20_points.ply"),
        help="Path to segmented PLY (with per-vertex RGB colors).",
    )
    parser.add_argument("--k", type=int, default=20, help="Number of clusters to iterate over.")
    args = parser.parse_args()

    if not os.path.exists(args.ply):
        raise FileNotFoundError(f"Missing PLY: {args.ply}")

    points, rgb255 = load_ply_xyz_rgb(args.ply)
    labels = labels_from_rgb255(rgb255, args.k)

    unknown = int(np.sum(labels < 0))
    if unknown > 0:
        print(
            f"[WARN] {unknown} / {labels.shape[0]} points are labeled as -1 (unknown/background).\n"
            "       This is fine if black/background exists; otherwise your PLY may contain colors not used for clusters."
        )

    for cid in range(args.k):
        print(f"[INFO] Showing overlay for cluster {cid} ...")
        show_cluster_overlay(points, rgb255, labels, cid)

    print("[DONE]")


if __name__ == "__main__":
    main()
