#!/usr/bin/env python3
import os
import json
import numpy as np
import copy
import cv2
import open3d as o3d

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.join(THIS_DIR, "output_scene")
TRELLIS_DIR = os.path.join(THIS_DIR, "trellis")

VGGT_PLY_PATH = os.path.join(SCENE_DIR, "fused_model.ply")
TRELLIS_PLY_PATH = os.path.join(TRELLIS_DIR, "trellis.ply")
ALIGNED_OUTPUT_PATH = os.path.join(TRELLIS_DIR, "trellis_aligned_to_vggt.ply")

RENDER_H = 512
RENDER_W = 512

# -----------------------------------------------------------------------------
# 1. HELPER: RENDERING
# -----------------------------------------------------------------------------
def render_point_cloud(points, colors, w2c_4x4, K, H, W, save_path):
    # 1. World -> Camera
    ones = np.ones((len(points), 1), dtype=np.float32)
    pts_hom = np.hstack([points.astype(np.float32), ones])
    pts_cam = (w2c_4x4 @ pts_hom.T).T

    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    # Filter Z > 0
    valid_z = z > 0.01
    if not np.any(valid_z):
        return
    x, y, z = x[valid_z], y[valid_z], z[valid_z]
    cur_colors = colors[valid_z]

    # 2. Camera -> Pixel
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = (x * fx / z) + cx
    v = (y * fy / z) + cy

    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # 3. Filter valid pixels
    valid_px = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(valid_px):
        return
    u, v, z = u[valid_px], v[valid_px], z[valid_px]
    cur_colors = cur_colors[valid_px]

    # 4. Draw
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
def load_geometry_as_pcd(path, sample_count=100000):
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
# 3. GLOBAL REGISTRATION (RANSAC + FPFH)
# -----------------------------------------------------------------------------
def preprocess_point_cloud(pcd, voxel_size):
    """
    Downsamples, estimates normals, and computes FPFH features.
    """
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate Normals (radius = 2 * voxel)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    # Compute FPFH Features (radius = 5 * voxel)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    
    return pcd_down, pcd_fpfh

def get_robust_scale_center(pcd):
    """Robustly gets scale/center using Median to ignore outliers."""
    # Clean first
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pts = np.asarray(cl.points)
    if len(pts) == 0: pts = np.asarray(pcd.points)

    center = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    scale = np.median(dists)
    return center, scale, cl

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    RANSAC Registration: Matches features to find rough alignment.
    """
    distance_threshold = voxel_size * 1.5
    print("   -> Running RANSAC...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result

def align_robust_global(source_pcd, target_pcd):
    print("--- Phase 1: Normalization ---")
    # 1. Robust Scaling/Centering
    s_center, s_scale, _ = get_robust_scale_center(source_pcd)
    t_center, t_scale, _ = get_robust_scale_center(target_pcd)

    scale_factor = t_scale / s_scale
    print(f"  Source Scale: {s_scale:.4f} | Target Scale: {t_scale:.4f}")
    print(f"  Scaling Factor: {scale_factor:.4f}")

    # Prepare Source (Scale & Center locally, don't move to target yet)
    source_temp = copy.deepcopy(source_pcd)
    source_temp.translate(-s_center)
    source_temp.scale(scale_factor, center=(0,0,0))
    
    # Prepare Target (Center at 0 for registration, move back later)
    target_temp = copy.deepcopy(target_pcd)
    target_temp.translate(-t_center)

    print("--- Phase 2: Feature Extraction ---")
    # Voxel size is critical. We define it relative to the object size.
    voxel_size = t_scale / 15.0  # Heuristic: Object size / 15
    print(f"  Voxel Size: {voxel_size:.4f}")

    s_down, s_fpfh = preprocess_point_cloud(source_temp, voxel_size)
    t_down, t_fpfh = preprocess_point_cloud(target_temp, voxel_size)

    print("--- Phase 3: Global Registration (RANSAC) ---")
    result_ransac = execute_global_registration(s_down, t_down, s_fpfh, t_fpfh, voxel_size)
    print(f"  RANSAC Fitness: {result_ransac.fitness:.4f}")

    print("--- Phase 4: Local Refinement (ICP) ---")
    result_icp = o3d.pipelines.registration.registration_icp(
        s_down, t_down, voxel_size * 0.4, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(f"  ICP Fitness:    {result_icp.fitness:.4f}")

    # Apply Transformations
    # 1. Apply the rotation/translation found by RANSAC/ICP (which happened at origin)
    source_temp.transform(result_icp.transformation)
    
    # 2. Move everything to the actual Target Center position
    source_temp.translate(t_center)

    return source_temp

# -----------------------------------------------------------------------------
# 4. MAIN
# -----------------------------------------------------------------------------
def main():
    if not os.path.exists(VGGT_PLY_PATH) or not os.path.exists(TRELLIS_PLY_PATH):
        print("Error: Input files missing.")
        return

    print("--- 1. Loading ---")
    target = load_geometry_as_pcd(VGGT_PLY_PATH)
    source = load_geometry_as_pcd(TRELLIS_PLY_PATH)

    print("\n--- 2. Global Alignment ---")
    aligned_trellis = align_robust_global(source, target)
    
    o3d.io.write_point_cloud(ALIGNED_OUTPUT_PATH, aligned_trellis)
    print(f"Saved aligned model: {ALIGNED_OUTPUT_PATH}")

    print("\n--- 3. Rendering ---")
    points = np.asarray(aligned_trellis.points)
    colors = np.asarray(aligned_trellis.colors)
    
    # Fallback Grey Color
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
        K = np.array(cam_data['intrinsics'], dtype=np.float32)
        
        base_name = cf.replace("_cam.json", "")
        out_path = os.path.join(TRELLIS_DIR, f"{base_name}_trellis.png")
        
        est_W = int(K[0, 2] * 2)
        est_H = int(K[1, 2] * 2)
        H_r = est_H if est_H > 64 else RENDER_H
        W_r = est_W if est_W > 64 else RENDER_W

        render_point_cloud(points, colors, w2c, K, H_r, W_r, out_path)
        print(f"  Rendered: {out_path}")

if __name__ == "__main__":
    main()