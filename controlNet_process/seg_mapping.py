import os
import json
import numpy as np
import cv2
import open3d as o3d
import glob
import re

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input Folders
SCENE_DIR = os.path.join(BASE_DIR, "vggt_output")
SEG_DIR = os.path.join(BASE_DIR, "segmentations")
OUTPUT_DIR = os.path.join(BASE_DIR, "final_overlays")

# PartField geometry (RE-ORDERED / CANONICAL)
PARTFIELD_COORDS = os.path.join(
    BASE_DIR,
    "PartField",
    "exp_results",
    "partfield_features",
    "single_inference",
    "fused_model",
    "coords.npy"
)

# 1. Mask Expansion (Pixels)
TOLERANCE = 1

# 2. Occlusion Threshold (Depth Units)
OCCLUSION_THRESHOLD = 0.05

COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
]

# -----------------------------------------------------------------------------
# MATH HELPERS
# -----------------------------------------------------------------------------
def get_scaled_intrinsics(K_orig, src_w, src_h, target_w, target_h):
    scale_x = target_w / src_w
    scale_y = target_h / src_h
    K_new = K_orig.copy()
    K_new[0, 0] *= scale_x; K_new[0, 2] *= scale_x
    K_new[1, 1] *= scale_y; K_new[1, 2] *= scale_y
    return K_new

def project_points(points, w2c_4x4, K, H, W):
    ones = np.ones((len(points), 1))
    pts_hom = np.hstack([points, ones])
    pts_cam = (w2c_4x4 @ pts_hom.T).T

    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid_z = z > 0.01
    
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    
    u[valid_z] = (x[valid_z] * fx / z[valid_z]) + cx
    v[valid_z] = (y[valid_z] * fy / z[valid_z]) + cy

    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    valid_px = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    return u, v, z, valid_px

def render_painter_algo(u, v, z, valid_mask, colors, H, W):
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = z[valid_mask]
    c_valid = colors[valid_mask]

    sort_idx = np.argsort(-z_valid)
    uu = u_valid[sort_idx]
    vv = v_valid[sort_idx]
    cc = (c_valid[sort_idx] * 255).astype(np.uint8)

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    cc_bgr = cc[:, [2, 1, 0]]
    canvas[vv, uu] = cc_bgr
    return canvas

def compute_depth_buffer(u, v, z, valid_mask, H, W):
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = z[valid_mask]
    np.minimum.at(depth_buffer, (v_valid, u_valid), z_valid)
    return depth_buffer

def save_labeled_ply(points, u, v, z, valid_mask, mask_paths, view_id, H, W):
    cloud_colors = np.zeros((len(points), 3), dtype=np.float64)

    depth_buffer = compute_depth_buffer(u, v, z, valid_mask, H, W)

    valid_indices = np.where(valid_mask)[0]
    u_val = u[valid_indices]
    v_val = v[valid_indices]
    z_val = z[valid_indices]

    surface_z = depth_buffer[v_val, u_val]
    is_visible = z_val <= (surface_z + OCCLUSION_THRESHOLD)

    visible_indices = valid_indices[is_visible]
    u_vis = u_val[is_visible]
    v_vis = v_val[is_visible]

    for i, mpath in enumerate(mask_paths):
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        if TOLERANCE > 0:
            kernel = np.ones((TOLERANCE*2+1, TOLERANCE*2+1), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        b, g, r = COLORS[i % len(COLORS)]
        rgb = np.array([r, g, b]) / 255.0

        mask_vals = mask[v_vis, u_vis]
        hits = mask_vals > 0

        if np.any(hits):
            cloud_colors[visible_indices[hits]] = rgb

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    pcd_out.colors = o3d.utility.Vector3dVector(cloud_colors)

    save_path = os.path.join(OUTPUT_DIR, f"{view_id}_labeled.ply")
    o3d.io.write_point_cloud(save_path, pcd_out)

def overlay_masks(base_img, mask_paths):
    overlay = base_img.copy()
    alpha = 0.5
    for i, mpath in enumerate(mask_paths):
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != base_img.shape[:2]:
            mask = cv2.resize(mask, (base_img.shape[1], base_img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        color = COLORS[i % len(COLORS)]
        colored = np.zeros_like(base_img)
        colored[:] = color
        binary = mask > 0
        overlay[binary] = cv2.addWeighted(
            overlay[binary], 1 - alpha, colored[binary], alpha, 0)
    return overlay

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    if not os.path.exists(PARTFIELD_COORDS):
        raise RuntimeError("Missing PartField coords.npy")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 🔹 LOAD PARTFIELD GEOMETRY (RE-ORDERED)
    points = np.load(PARTFIELD_COORDS)

    # 🔹 FORCE BLACK INPUT CLOUD
    orig_colors = np.zeros((len(points), 3), dtype=np.float64)

    cam_files = glob.glob(os.path.join(SCENE_DIR, "*_cam.json"))
    if not cam_files:
        return

    for cam_json_path in cam_files:
        filename = os.path.basename(cam_json_path)
        match = re.search(r"(view_\d+)_cam\.json", filename)
        if not match:
            continue
        view_id = match.group(1)

        seg_folder = os.path.join(SEG_DIR, view_id)
        if not os.path.exists(seg_folder):
            continue
        mask_paths = glob.glob(os.path.join(seg_folder, "*_mask.png"))
        if not mask_paths:
            continue

        temp_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
        H, W = temp_mask.shape

        with open(cam_json_path, "r") as f:
            cam_data = json.load(f)

        w2c = np.array(cam_data["extrinsics_w2c"])
        K_orig = np.array(cam_data["intrinsics"])

        src_w = K_orig[0, 2] * 2
        src_h = K_orig[1, 2] * 2

        K_scaled = get_scaled_intrinsics(K_orig, src_w, src_h, W, H)

        u, v, z, valid_mask = project_points(points, w2c, K_scaled, H, W)

        render_img = render_painter_algo(
            u, v, z, valid_mask, orig_colors, H, W
        )

        final_comp = overlay_masks(render_img, mask_paths)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{view_id}_overlay.png"), final_comp)

        save_labeled_ply(points, u, v, z, valid_mask, mask_paths, view_id, H, W)

if __name__ == "__main__":
    main()
