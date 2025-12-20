import os
import json
import numpy as np
import cv2
import open3d as o3d
import glob
import re

# this code takes the per-view segmentation images and map them to 3D
# we first produce a 3D segmentation for each view, and then combine them
# this has nothing to do with the clusters. But we do use partfield output shape as the 3D input

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")

SCENE_DIR = os.path.join(SKETCH_ROOT, "3d_reconstruction")
SEG_DIR   = os.path.join(SKETCH_ROOT, "segmentation")

# (1) output folder should be partfield_overlay
OUTPUT_DIR = os.path.join(SKETCH_ROOT, "partfield_overlay")

# (2) input ply file should be clustering_k20_points.ply from the same folder
PLY_PATH   = os.path.join(SCENE_DIR, "clustering_k20_points.ply")

TOLERANCE = 1
OCCLUSION_THRESHOLD = 0.05

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

    sort_idx = np.argsort(-z_valid)  # far -> near
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

# -----------------------------------------------------------------------------
# LABEL / COLOR HELPERS
# -----------------------------------------------------------------------------
def distinct_colors_bgr(n):
    """
    Deterministic palette via HSV sweep.
    Returns list of (B,G,R) uint8.
    """
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        h = int((i * 179) / max(1, n))  # OpenCV hue range [0,179]
        s = 200
        v = 255
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors

def mask_label_from_path(mpath):
    # "chair_leg_mask.png" -> "chair_leg"
    return os.path.basename(mpath).replace("_mask.png", "")

# -----------------------------------------------------------------------------
# PER-VIEW SAVE (kept, but now uses global label->color)
# -----------------------------------------------------------------------------
def save_labeled_ply_and_json(points, u, v, z, valid_mask, mask_paths, view_id, H, W,
                             label_to_id, label_to_color_bgr):
    """
    Saves FULL cloud colored by THIS view's masks, plus JSON mapping label->color.
    Uses global label_to_color_bgr so colors are consistent across views.
    """
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

    semantic_metadata = {}

    for mpath in mask_paths:
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        label = mask_label_from_path(mpath)
        if label not in label_to_id:
            continue

        if TOLERANCE > 0:
            k = TOLERANCE * 2 + 1
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        b, g, r = label_to_color_bgr[label]
        rgb_color = np.array([r, g, b]) / 255.0

        semantic_metadata[label] = {
            "label_id": int(label_to_id[label]),
            "color_bgr": [int(b), int(g), int(r)],
            "color_rgb_norm": [float(r/255.0), float(g/255.0), float(b/255.0)],
        }

        mask_vals = mask[v_vis, u_vis]
        hits = mask_vals > 0
        if np.any(hits):
            cloud_colors[visible_indices[hits]] = rgb_color

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    pcd_out.colors = o3d.utility.Vector3dVector(cloud_colors)

    ply_save_path = os.path.join(OUTPUT_DIR, f"{view_id}_labeled.ply")
    o3d.io.write_point_cloud(ply_save_path, pcd_out)

    json_save_path = os.path.join(OUTPUT_DIR, f"{view_id}_labels.json")
    with open(json_save_path, "w") as f:
        json.dump(semantic_metadata, f, indent=2)

def overlay_masks(base_img, mask_paths, label_to_color_bgr):
    overlay = base_img.copy()
    alpha = 0.5
    for mpath in mask_paths:
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != base_img.shape[:2]:
            mask = cv2.resize(mask, (base_img.shape[1], base_img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        label = mask_label_from_path(mpath)
        color = label_to_color_bgr.get(label, (255, 255, 255))

        colored_mask = np.zeros_like(base_img)
        colored_mask[:] = color
        binary_mask = mask > 0
        overlay[binary_mask] = cv2.addWeighted(
            overlay[binary_mask], 1 - alpha, colored_mask[binary_mask], alpha, 0
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)

    return overlay

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(PLY_PATH):
        print(f"Error: Could not find fused model at {PLY_PATH}")
        return

    print(f"Loading 3D model: {PLY_PATH}")
    pcd = o3d.io.read_point_cloud(PLY_PATH)
    points = np.asarray(pcd.points)

    # Treat input PLY as all-black; colors come only from our mapping step.
    orig_colors = np.zeros((len(points), 3), dtype=np.float64)


    # Cameras
    cam_files = sorted(glob.glob(os.path.join(SCENE_DIR, "*_cam.json")))
    if not cam_files:
        print(f"Error: No camera JSON files found in {SCENE_DIR}")
        return

    # -------------------------------------------------------------------------
    # 0) Build global label set across ALL views (for consistent colors + voting)
    # -------------------------------------------------------------------------
    all_masks = glob.glob(os.path.join(SEG_DIR, "view_*", "*_mask.png"))
    if not all_masks:
        print(f"Error: No masks found under {SEG_DIR}/view_*/")
        return

    labels = sorted(set(mask_label_from_path(p) for p in all_masks))
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    palette_bgr = distinct_colors_bgr(len(labels))
    label_to_color_bgr = {labels[i]: palette_bgr[i] for i in range(len(labels))}

    # Global JSON mapping (label <-> color)
    label_color_map = {
        "labels_in_order": labels,
        "label_to_id": label_to_id,
        "label_to_color_bgr": {lab: list(label_to_color_bgr[lab]) for lab in labels},
        "label_to_color_rgb": {lab: [label_to_color_bgr[lab][2],
                                     label_to_color_bgr[lab][1],
                                     label_to_color_bgr[lab][0]] for lab in labels},
        "color_bgr_to_label": {f"{b},{g},{r}": lab for lab, (b, g, r) in label_to_color_bgr.items()},
    }
    with open(os.path.join(OUTPUT_DIR, "label_color_map.json"), "w") as f:
        json.dump(label_color_map, f, indent=2)

    # Voting matrix
    N = len(points)
    L = len(labels)
    votes = np.zeros((N, L), dtype=np.uint16)

    # -------------------------------------------------------------------------
    # 1) Per-view processing + vote accumulation
    # -------------------------------------------------------------------------
    for cam_json_path in cam_files:
        filename = os.path.basename(cam_json_path)
        match = re.search(r"(view_\d+)_cam\.json", filename)
        if not match:
            continue
        view_id = match.group(1)

        seg_folder = os.path.join(SEG_DIR, view_id)
        if not os.path.exists(seg_folder):
            print(f"Warning: No segmentation folder for {view_id}")
            continue

        mask_paths = sorted(glob.glob(os.path.join(seg_folder, "*_mask.png")))
        if not mask_paths:
            print(f"Warning: No masks in {seg_folder}")
            continue

        print(f"Processing {view_id}...")

        temp_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
        if temp_mask is None:
            continue
        target_h, target_w = temp_mask.shape

        # Try to infer source image size (verification render), else fall back to 2*cx,2*cy
        verif_img_path = os.path.join(SCENE_DIR, f"{view_id}_verification_render.png")
        source_h, source_w = 0, 0
        if os.path.exists(verif_img_path):
            v_img = cv2.imread(verif_img_path)
            if v_img is not None:
                source_h, source_w = v_img.shape[:2]

        with open(cam_json_path, "r") as f:
            cam_data = json.load(f)

        w2c = np.array(cam_data["extrinsics_w2c"])
        K_orig = np.array(cam_data["intrinsics"])

        if source_w == 0:
            source_w = K_orig[0, 2] * 2
            source_h = K_orig[1, 2] * 2

        K_scaled = get_scaled_intrinsics(K_orig, source_w, source_h, target_w, target_h)

        # Project
        u, v, z, valid_mask = project_points(points, w2c, K_scaled, target_h, target_w)

        # Visibility
        depth_buffer = compute_depth_buffer(u, v, z, valid_mask, target_h, target_w)

        valid_indices = np.where(valid_mask)[0]
        u_val = u[valid_indices]
        v_val = v[valid_indices]
        z_val = z[valid_indices]

        surface_z = depth_buffer[v_val, u_val]
        is_visible = z_val <= (surface_z + OCCLUSION_THRESHOLD)

        visible_indices = valid_indices[is_visible]
        u_vis = u_val[is_visible]
        v_vis = v_val[is_visible]

        # Accumulate votes for each label
        for mpath in mask_paths:
            mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            if TOLERANCE > 0:
                k = TOLERANCE * 2 + 1
                kernel = np.ones((k, k), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

            label = mask_label_from_path(mpath)
            lid = label_to_id.get(label, None)
            if lid is None:
                continue

            mask_vals = mask[v_vis, u_vis]
            hits = mask_vals > 0
            if np.any(hits):
                votes[visible_indices[hits], lid] += 1

        # 2D overlay (unchanged idea, but consistent colors now)
        render_img = render_painter_algo(u, v, z, valid_mask, orig_colors, target_h, target_w)
        final_comp = overlay_masks(render_img, mask_paths, label_to_color_bgr)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{view_id}_overlay.png"), final_comp)

        # Per-view labeled PLY + per-view JSON (optional but kept)
        save_labeled_ply_and_json(
            points, u, v, z, valid_mask, mask_paths, view_id,
            target_h, target_w,
            label_to_id, label_to_color_bgr
        )

    # -------------------------------------------------------------------------
    # 2) Merge to one segmentation by voting
    # -------------------------------------------------------------------------
    vote_sums = votes.sum(axis=1)
    best = np.argmax(votes, axis=1)
    has_label = vote_sums > 0

    merged_colors = np.zeros((N, 3), dtype=np.float64)  # default black
    palette_rgb01 = np.array([[bgr[2], bgr[1], bgr[0]] for bgr in palette_bgr], dtype=np.float64) / 255.0
    merged_colors[has_label] = palette_rgb01[best[has_label]]

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(points)
    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    merged_ply_path = os.path.join(OUTPUT_DIR, "merged_labeled.ply")
    o3d.io.write_point_cloud(merged_ply_path, merged_pcd)

    # Optional: save per-point label ids too (handy for later clustering/debug)
    merged_ids_path = os.path.join(OUTPUT_DIR, "merged_label_ids.npy")
    merged_label_ids = np.full((N,), -1, dtype=np.int32)
    merged_label_ids[has_label] = best[has_label].astype(np.int32)
    np.save(merged_ids_path, merged_label_ids)

    print(f"[OK] merged: {merged_ply_path}")
    print(f"[OK] label map: {os.path.join(OUTPUT_DIR, 'label_color_map.json')}")
    print(f"[OK] label ids: {merged_ids_path}")

if __name__ == "__main__":
    main()
