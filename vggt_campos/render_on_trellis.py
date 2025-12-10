#!/usr/bin/env python3
import os
import json
import numpy as np
import copy
import cv2
import open3d as o3d

# -----------------------------------------------------------------------------
# CONFIGURATION   (fixed to match your previous working code)
# -----------------------------------------------------------------------------
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))

# VGGT output folder
SCENE_DIR   = os.path.join(THIS_DIR, "output_scene")

# Trellis folder (has trellis.ply)
TRELLIS_DIR = os.path.join(THIS_DIR, "trellis")

VGGT_PLY_PATH       = os.path.join(SCENE_DIR, "fused_model_clean.ply")
TRELLIS_PLY_PATH    = os.path.join(TRELLIS_DIR, "trellis.ply")
ALIGNED_OUTPUT_PATH = os.path.join(TRELLIS_DIR, "trellis_aligned_to_vggt.ply")
RENDER_H = 512
RENDER_W = 512

# Max points used for registration (RANSAC + ICP). Full cloud is only transformed once at the end.
MAX_REG_POINTS = 15000

# -----------------------------------------------------------------------------
# 1. HELPER: RENDERING
# -----------------------------------------------------------------------------
def render_point_cloud(points, colors, w2c_4x4, K, H, W, save_path):
    ones = np.ones((len(points), 1), dtype=np.float32)
    pts_hom = np.hstack([points.astype(np.float32), ones])
    pts_cam = (w2c_4x4 @ pts_hom.T).T

    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    valid_z = z > 0.01
    if not np.any(valid_z):
        print(f"[WARN] No valid points in front of camera for {save_path}")
        return
    x, y, z = x[valid_z], y[valid_z], z[valid_z]
    cur_colors = colors[valid_z]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = (x * fx / z) + cx
    v = (y * fy / z) + cy

    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    valid_px = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(valid_px):
        print(f"[WARN] All projected pixels outside image for {save_path}")
        return
    u, v, z = u[valid_px], v[valid_px], z[valid_px]
    cur_colors = cur_colors[valid_px]

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    sort_idx = np.argsort(-z)
    uu = u[sort_idx]
    vv = v[sort_idx]
    cc = (cur_colors[sort_idx] * 255).astype(np.uint8)

    canvas[vv, uu] = cc
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

# -----------------------------------------------------------------------------
# 2. HELPER: GEOMETRY LOADING
# -----------------------------------------------------------------------------
def load_geometry_as_pcd(path, sample_count=120000):
    try:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) > 0:
            print(f"[{os.path.basename(path)}] Loaded Mesh. Sampling {sample_count} points...")
            pcd = mesh.sample_points_poisson_disk(number_of_points=sample_count)
            return pcd
    except:
        pass
    print(f"[{os.path.basename(path)}] Loaded PointCloud.")
    return o3d.io.read_point_cloud(path)

# -----------------------------------------------------------------------------
# 3. GLOBAL REGISTRATION (RANSAC + FPFH) WITH HARD POINT LIMIT
# -----------------------------------------------------------------------------
def preprocess_point_cloud_fast(pcd, voxel_size):
    """
    Downsample, then cap to MAX_REG_POINTS, estimate normals, compute FPFH.
    This is what RANSAC + ICP see. Full-res cloud is transformed only once later.
    """
    # 1) Voxel downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    pts = np.asarray(pcd_down.points)
    if pts.shape[0] > MAX_REG_POINTS:
        # Deterministic sub-sampling: take evenly spaced indices
        step = int(np.ceil(pts.shape[0] / MAX_REG_POINTS))
        idx = np.arange(0, pts.shape[0], step)
        pcd_down = pcd_down.select_by_index(idx)
        print(f"  [preprocess] clipped {pts.shape[0]} -> {len(pcd_down.points)} points for registration")

    # 2) Normals
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # 3) FPFH
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )

    return pcd_down, pcd_fpfh

def get_scale_center_no_clean(pcd):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise ValueError("Empty point cloud")
    center = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    scale = np.median(dists)
    return center, scale

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print("   -> Running RANSAC (feature-based, on capped clouds)...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result

def align_robust_global_fast(source_pcd_full, target_pcd_full):
    print("--- Phase 1: Normalization (no cleanup) ---")
    s_center, s_scale = get_scale_center_no_clean(source_pcd_full)
    t_center, t_scale = get_scale_center_no_clean(target_pcd_full)

    scale_factor = t_scale / s_scale
    print(f"  Source Scale: {s_scale:.4f} | Target Scale: {t_scale:.4f}")
    print(f"  Scaling Factor: {scale_factor:.4f}")

    # Work copies for *registration only* (downsampled later)
    source_reg = copy.deepcopy(source_pcd_full)
    source_reg.translate(-s_center)
    source_reg.scale(scale_factor, center=(0, 0, 0))

    target_reg = copy.deepcopy(target_pcd_full)
    target_reg.translate(-t_center)

    print("--- Phase 2: Feature Extraction (capped) ---")
    voxel_size = t_scale / 15.0
    print(f"  Voxel Size: {voxel_size:.4f}")

    s_down, s_fpfh = preprocess_point_cloud_fast(source_reg, voxel_size)
    t_down, t_fpfh = preprocess_point_cloud_fast(target_reg, voxel_size)
    print(f"  s_down points: {len(s_down.points)}, t_down points: {len(t_down.points)}")

    print("--- Phase 3: Global Registration (RANSAC) ---")
    result_ransac = execute_global_registration(s_down, t_down, s_fpfh, t_fpfh, voxel_size)
    print(f"  RANSAC Fitness: {result_ransac.fitness:.4f}")

    print("--- Phase 4: Local Refinement (ICP on capped clouds) ---")
    result_icp = o3d.pipelines.registration.registration_icp(
        s_down, t_down,
        voxel_size * 0.4,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(f"  ICP Fitness:    {result_icp.fitness:.4f}")

    # Now apply the *same* transform to the full-resolution source
    aligned_full = copy.deepcopy(source_pcd_full)
    aligned_full.translate(-s_center)
    aligned_full.scale(scale_factor, center=(0, 0, 0))
    aligned_full.transform(result_icp.transformation)
    aligned_full.translate(t_center)

    return aligned_full

# -----------------------------------------------------------------------------
# 4. MAIN
# -----------------------------------------------------------------------------
def main():
    if not os.path.exists(VGGT_PLY_PATH) or not os.path.exists(TRELLIS_PLY_PATH):
        print("Error: Input files missing.")
        print(f"  VGGT_PLY_PATH:    {VGGT_PLY_PATH}")
        print(f"  TRELLIS_PLY_PATH: {TRELLIS_PLY_PATH}")
        return

    print("--- 1. Loading ---")
    target = load_geometry_as_pcd(VGGT_PLY_PATH)
    source = load_geometry_as_pcd(TRELLIS_PLY_PATH)

    print("\n--- 2. Global 3D Alignment (RANSAC + ICP, capped points) ---")
    aligned_trellis = align_robust_global_fast(source, target)

    o3d.io.write_point_cloud(ALIGNED_OUTPUT_PATH, aligned_trellis)
    print(f"Saved aligned model: {ALIGNED_OUTPUT_PATH}")

    print("\n--- 3. Rendering ---")
    points = np.asarray(aligned_trellis.points)
    colors = np.asarray(aligned_trellis.colors)

    if colors.shape[0] != points.shape[0] or colors.shape[0] == 0:
        colors = np.ones_like(points) * 0.5

    cam_files = sorted([f for f in os.listdir(SCENE_DIR) if f.endswith('_cam.json')])

    if not cam_files:
        print("No cameras found.")
        return

    for cf in cam_files:
        json_path = os.path.join(SCENE_DIR, cf)
        with open(json_path, 'r') as f:
            cam_data = json.load(f)

        w2c = np.array(cam_data['extrinsics_w2c'], dtype=np.float32)
        K   = np.array(cam_data['intrinsics'],    dtype=np.float32)

        base_name = cf.replace("_cam.json", "")
        out_path = os.path.join(TRELLIS_DIR, f"{base_name}_trellis.png")

        est_W = int(K[0, 2] * 2)
        est_H = int(K[1, 2] * 2)
        H_r   = est_H if est_H > 64 else RENDER_H
        W_r   = est_W if est_W > 64 else RENDER_W

        render_point_cloud(points, colors, w2c, K, H_r, W_r, out_path)
        print(f"  Rendered: {out_path}")

if __name__ == "__main__":
    main()
