#!/usr/bin/env python3
import os
import sys
import numpy as np
import open3d as o3d


# -----------------------------------------------------------------------------
# CONFIG (edit if needed)
# -----------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

SCENE_DIR = os.path.join(ROOT_DIR, "sketch", "3d_reconstruction")
CLUSTERS_DIR = os.path.join(ROOT_DIR, "sketch", "clusters")

FUSED_PLY_PATH = os.path.join(SCENE_DIR, "fused_model.ply")

# Choose which .npy to visualize:
#   - final_cluster_ids.npy: your post-processed ids (recommended)
#   - clustering_k20.npy: original clusters
USE_FINAL = True

FINAL_NPY_PATH = os.path.join(CLUSTERS_DIR, "final_cluster_ids.npy")
ORIG_NPY_PATH  = os.path.join(SCENE_DIR, "clustering_k20.npy")

# Visualization params
BG_GRAY = np.array([0.70, 0.70, 0.70], dtype=np.float64)
CL_RED  = np.array([1.00, 0.15, 0.15], dtype=np.float64)

# Skip negative clusters by default
INCLUDE_NEGATIVE = False

# If True, show clusters in descending size (largest first)
SORT_BY_SIZE_DESC = True

# Downsample background for speed (cluster points always kept)
# Set to None to disable. Example: 5 means keep ~20% of background points.
BACKGROUND_STRIDE = 5

# Add an AABB around the highlighted cluster
SHOW_AABB = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _unique_count(points_xyz: np.ndarray) -> int:
    if points_xyz.shape[0] == 0:
        return 0
    # exact uniqueness (float) can be unstable; still useful as a sanity check
    return np.unique(points_xyz, axis=0).shape[0]

def _unique_count_rounded(points_xyz: np.ndarray, decimals: int = 6) -> int:
    if points_xyz.shape[0] == 0:
        return 0
    pts = np.round(points_xyz, decimals=decimals)
    return np.unique(pts, axis=0).shape[0]

def _make_cloud(pts: np.ndarray, color_rgb: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    cols = np.repeat(color_rgb[None, :], pts.shape[0], axis=0)
    p.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    return p

def _make_aabb_lineset(pts: np.ndarray, color=(0.1, 0.7, 0.1)) -> o3d.geometry.LineSet:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    aabb = pc.get_axis_aligned_bounding_box()
    aabb.color = color
    return o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    npy_path = FINAL_NPY_PATH if USE_FINAL else ORIG_NPY_PATH

    print(f"[LOAD] Point cloud: {FUSED_PLY_PATH}")
    if not os.path.exists(FUSED_PLY_PATH):
        raise FileNotFoundError(FUSED_PLY_PATH)

    pcd = o3d.io.read_point_cloud(FUSED_PLY_PATH)
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        raise RuntimeError("Empty point cloud.")

    print(f"[LOAD] Cluster ids: {npy_path}")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(npy_path)

    cluster_ids = np.load(npy_path).reshape(-1)
    if cluster_ids.shape[0] != pts.shape[0]:
        raise RuntimeError(
            f"Cluster ids length {cluster_ids.shape[0]} != points {pts.shape[0]}.\n"
            "The .npy must align point-for-point with fused_model.ply."
        )

    unique_ids = np.unique(cluster_ids)
    if not INCLUDE_NEGATIVE:
        unique_ids = unique_ids[unique_ids >= 0]

    unique_ids = [int(x) for x in unique_ids.tolist()]

    # Precompute sizes for sorting
    sizes = {cid: int(np.sum(cluster_ids == cid)) for cid in unique_ids}
    if SORT_BY_SIZE_DESC:
        unique_ids.sort(key=lambda c: sizes[c], reverse=True)
    else:
        unique_ids.sort()

    print(f"[INFO] Total points: {pts.shape[0]}")
    print(f"[INFO] Clusters to visualize: {len(unique_ids)} (include_negative={INCLUDE_NEGATIVE})")
    print("[INFO] Controls: close window to move to next cluster.")

    # Background indices (optionally subsampled for speed)
    if BACKGROUND_STRIDE is None or BACKGROUND_STRIDE <= 1:
        bg_keep = np.arange(pts.shape[0], dtype=np.int64)
    else:
        bg_keep = np.arange(0, pts.shape[0], int(BACKGROUND_STRIDE), dtype=np.int64)

    # Iterate clusters
    for i, cid in enumerate(unique_ids):
        idx = np.where(cluster_ids == cid)[0]
        n = int(idx.size)
        if n == 0:
            continue

        cluster_pts = pts[idx]

        # Useful diagnostics for your "unique_points=1" issue:
        uniq_raw = _unique_count(cluster_pts)
        uniq_r6 = _unique_count_rounded(cluster_pts, decimals=6)

        print("\n" + "-" * 80)
        print(f"[CLUSTER] {i+1}/{len(unique_ids)}  cid={cid}  points={n}")
        print(f"          unique_raw={uniq_raw}  unique_rounded_1e-6={uniq_r6}")

        # If it collapses to 1 unique point after rounding, print the representative point
        if uniq_r6 <= 3:
            rep = np.round(cluster_pts[0], 9).tolist()
            print(f"          [WARN] Very low unique_rounded; example point: {rep}")

        # Build geometries:
        # Background cloud (gray) + cluster cloud (red)
        # Background excludes cluster points to make highlight clearer.
        bg_mask = np.ones((pts.shape[0],), dtype=bool)
        bg_mask[idx] = False
        bg_idx = bg_keep[bg_mask[bg_keep]]

        bg_cloud = _make_cloud(pts[bg_idx], BG_GRAY)
        cl_cloud = _make_cloud(cluster_pts, CL_RED)

        geoms = [bg_cloud, cl_cloud]
        if SHOW_AABB and cluster_pts.shape[0] >= 1:
            geoms.append(_make_aabb_lineset(cluster_pts))

        # Show
        title = f"cluster cid={cid} (points={n}, unique_r6={uniq_r6})"
        o3d.visualization.draw_geometries(geoms, window_name=title)

    print("[DONE] Finished visualizing all clusters.")


if __name__ == "__main__":
    main()
