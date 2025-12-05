#!/usr/bin/env python3
"""
verify_and_overlay.py

1. Reads 'independent_view_fits/all_cameras.json' and the .ply file.
2. Re-renders the views (verification step) to get the base object shape.
3. Looks for "./segmentations/view_{x}/" relative to where this script is run.
4. Overlays masks (e.g., "backrest_1_mask.png") with unique colors in 2D.
5. Projects the 2D overlap (rendered silhouette ∩ segmentation) back to 3D and
   saves per-view colored .ply/.ply mesh files, with
   - each component having its own color (even if labels repeat),
   - slight tolerance via 2D dilation to color more neighboring points.
"""

import json
import math
import numpy as np
import cv2
import open3d as o3d
from open3d.visualization import rendering
from pathlib import Path
import sys

# ---------------- CONFIGURATION ----------------

# Relative paths (assuming you run this script from the root folder)
PLY_PATH       = Path("trellis_outputs/0_trellis_gaussian.ply")
JSON_PATH      = Path("independent_view_fits/all_cameras.json")
SEG_DIR        = Path("segmentations")
OUTPUT_DIR     = Path("final_overlays")
IMG_SIZE       = 160

# Camera intrinsics (must match rendering)
FOV_DEG = 40.0
ASPECT  = 1.0  # square images

# Tolerance for mask → 3D mapping
DILATION_KERNEL_SIZE = 3     # 3x3 kernel
DILATION_ITERATIONS  = 1     # expand slightly

# ---------------- COLOR MANAGEMENT ----------------

# Explicit bright colors for visibility (B, G, R)
DISTINCT_COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (0, 165, 255),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (128, 128, 0),  # Olive
    (128, 128, 128) # Grey
]

component_color_map = {}
component_color_index = 0

def get_color_for_component(component_id: str):
    """
    Assigns a consistent color to a specific component instance, keyed by
    its full mask stem (e.g., 'leg_0_mask' or 'backrest_2_mask').

    This guarantees that:
    - Different components get different colors.
    - Even same label (e.g., leg_0 vs leg_1) get different colors.
    """
    global component_color_index
    if component_id not in component_color_map:
        component_color_map[component_id] = DISTINCT_COLORS[component_color_index % len(DISTINCT_COLORS)]
        component_color_index += 1
    return component_color_map[component_id]

# ---------------- HELPER FUNCTIONS (Rendering) ----------------

def sph_dir(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = math.cos(el) * math.sin(az)
    y = math.cos(el) * math.cos(az)
    z = math.sin(el)
    return np.array([x, y, z], dtype=float)

def load_geometry(path: Path):
    if not path.exists():
        print(f"❌ Error: PLY file not found at {path.resolve()}")
        sys.exit(1)
    geom = o3d.io.read_point_cloud(str(path))
    if geom.is_empty():
        geom = o3d.io.read_triangle_mesh(str(path))
        geom.compute_vertex_normals()
    return geom

def compute_normalization_params(mask: np.ndarray, size: int):
    """
    From a raw binary mask (0/1), compute the normalization parameters
    used by normalize_mask (crop bbox, scale, and placement).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    h_box = y_max - y_min + 1
    w_box = x_max - x_min + 1

    target_size = int(size * 0.85)
    scale      = target_size / max(h_box, w_box)
    new_h      = int(h_box * scale)
    new_w      = int(w_box * scale)

    start_y = (size - new_h) // 2
    start_x = (size - new_w) // 2

    return {
        "x_min":   x_min,
        "y_min":   y_min,
        "h_box":   h_box,
        "w_box":   w_box,
        "scale":   scale,
        "start_x": start_x,
        "start_y": start_y,
        "new_h":   new_h,
        "new_w":   new_w,
    }

def normalize_mask(mask: np.ndarray, size: int, params=None) -> (np.ndarray, dict):
    """
    Centers and scales the mask (85% fill) - same as search script.
    Returns (normalized_mask, params).
    """
    if params is None:
        params = compute_normalization_params(mask, size)

    if params is None:
        return np.zeros((size, size), dtype=np.uint8), None

    x_min   = params["x_min"]
    y_min   = params["y_min"]
    scale   = params["scale"]
    start_x = params["start_x"]
    start_y = params["start_y"]
    new_h   = params["new_h"]
    new_w   = params["new_w"]

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8), params

    y_max = ys.max()
    x_max = xs.max()

    crop = mask[y_min:y_max+1, x_min:x_max+1]

    # Resize cropped region
    resized = cv2.resize(
        (crop.astype(np.uint8) * 255),
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    return (canvas > 0).astype(np.uint8), params

def render_solid_silhouette(renderer, center, radius, az, el) -> np.ndarray:
    """Renders 3D object from specific view and returns a raw silhouette mask."""
    scene = renderer.scene
    direction = sph_dir(az, el)
    eye = center + radius * direction
    up = np.array([0.0, 0.0, 1.0])

    cam = scene.camera
    cam.set_projection(FOV_DEG, ASPECT, radius*0.01, radius*100.0,
                       rendering.Camera.FovType.Vertical)
    cam.look_at(center, eye, up)

    img_o3d = renderer.render_to_image()
    img = np.asarray(img_o3d)
    # Background is white; object is black due to material
    mask = (img[:, :, 0] < 200).astype(np.uint8)
    return mask

# ---------------- BACKPROJECTION HELPERS ----------------

def compute_camera_basis(center, radius, az_deg, el_deg):
    """
    Reconstruct the same camera pose as render_solid_silhouette:
      eye = center + radius * sph_dir(az, el)
      forward = center - eye
      up_world = (0, 0, 1)
    Returns eye, forward, right, up_cam.
    """
    direction = sph_dir(az_deg, el_deg)
    eye = center + radius * direction

    forward = center - eye
    norm_f = np.linalg.norm(forward)
    if norm_f < 1e-8:
        forward = np.array([0.0, 0.0, 1.0])
        norm_f = 1.0
    forward = forward / norm_f

    up_world = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, up_world)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        up_world = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(forward, up_world)
        right_norm = np.linalg.norm(right)
    right = right / right_norm

    up_cam = np.cross(right, forward)
    up_cam = up_cam / np.linalg.norm(up_cam)

    return eye, forward, right, up_cam

def project_points_to_pixels(points, eye, forward, right, up_cam,
                             img_w, img_h, fov_deg=FOV_DEG, aspect=ASPECT):
    """
    Perspective-project 3D points into image pixels (raw render space).

    Returns:
      - valid_mask: boolean array [N], True if point is in front of camera and within image bounds
      - u_coords, v_coords: integer pixel coordinates [N] in [0, W/H)
    """
    pts_rel = points - eye  # (N, 3)

    # Camera-space coords
    x_cam = np.dot(pts_rel, right)    # (N,)
    y_cam = np.dot(pts_rel, up_cam)   # (N,)
    z_cam = np.dot(pts_rel, forward)  # (N,)

    # Only points in front of the camera
    valid = z_cam > 1e-6
    if not np.any(valid):
        return valid, np.zeros_like(z_cam, int), np.zeros_like(z_cam, int)

    z_cam_valid = z_cam[valid]
    x_cam_valid = x_cam[valid]
    y_cam_valid = y_cam[valid]

    fov_rad = math.radians(fov_deg)
    half_h = math.tan(fov_rad / 2.0)
    half_w = half_h * aspect

    # Normalized device coordinates in [-1, 1]
    x_ndc = x_cam_valid / (z_cam_valid * half_w)
    y_ndc = y_cam_valid / (z_cam_valid * half_h)

    # Convert to pixel coordinates
    u = (x_ndc * 0.5 + 0.5) * img_w
    v = (1.0 - (y_ndc * 0.5 + 0.5)) * img_h  # flip y

    u = u.astype(np.int32)
    v = v.astype(np.int32)

    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

    full_valid = np.zeros_like(valid)
    full_valid_idx = np.where(valid)[0][in_bounds]

    full_valid[full_valid_idx] = True

    u_full = np.zeros_like(z_cam, dtype=np.int32)
    v_full = np.zeros_like(z_cam, dtype=np.int32)
    u_full[full_valid_idx] = u[in_bounds]
    v_full[full_valid_idx] = v[in_bounds]

    return full_valid, u_full, v_full

def apply_roll_to_pixels(u, v, roll_deg, img_w, img_h):
    """
    Apply the same 2D roll as:
      M = cv2.getRotationMatrix2D(center_pt, roll, 1.0)
      rotated = warpAffine(...)
    Here we transform the pixel coordinates directly.
    """
    center_pt = (img_w / 2.0, img_h / 2.0)
    M = cv2.getRotationMatrix2D(center_pt, roll_deg, 1.0)  # 2x3

    ones = np.ones_like(u, dtype=np.float32)
    pts = np.stack([u.astype(np.float32), v.astype(np.float32), ones], axis=-1)  # (N,3)
    uv_rot = pts @ M.T  # (N,2)

    u_rot = uv_rot[:, 0]
    v_rot = uv_rot[:, 1]

    return u_rot, v_rot

def raw_to_normalized_pixels(u_raw, v_raw, norm_params, img_size):
    """
    Map raw render-space pixel coordinates into normalized mask space,
    using the same transform as normalize_mask.
    """
    if norm_params is None:
        return u_raw.astype(np.float32), v_raw.astype(np.float32)

    x_min   = norm_params["x_min"]
    y_min   = norm_params["y_min"]
    scale   = norm_params["scale"]
    start_x = norm_params["start_x"]
    start_y = norm_params["start_y"]

    # Shift into cropped bbox
    x_box = u_raw.astype(np.float32) - float(x_min)
    y_box = v_raw.astype(np.float32) - float(y_min)

    # Scale
    x_scaled = x_box * scale
    y_scaled = y_box * scale

    # Place into full canvas
    u_norm = start_x + x_scaled
    v_norm = start_y + y_scaled

    return u_norm, v_norm

def precompute_projection(geom, center, radius, az, el, roll, img_size, norm_params):
    """
    For a given view, project all 3D points once into final 2D (normalized + rolled)
    coordinates. Returns:

      - pts: (N,3) array of 3D points
      - valid_mask: (N,) bool, True where projection is valid & in-bounds
      - u_pix, v_pix: (N,) int pixel coords in [0, img_size)
    """
    # 1. Extract geometry points
    if isinstance(geom, o3d.geometry.PointCloud):
        pts = np.asarray(geom.points)  # (N,3)
    elif isinstance(geom, o3d.geometry.TriangleMesh):
        pts = np.asarray(geom.vertices)  # (N,3)
    else:
        raise TypeError("geom must be PointCloud or TriangleMesh")

    N = len(pts)
    if N == 0:
        return pts, np.zeros(0, dtype=bool), np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    # 2. Camera basis
    eye, forward, right, up_cam = compute_camera_basis(center, radius, az, el)

    # 3. Project to raw render-space pixels
    valid, u_raw, v_raw = project_points_to_pixels(
        pts, eye, forward, right, up_cam,
        img_size, img_size, fov_deg=FOV_DEG, aspect=ASPECT
    )

    # 4. Map raw -> normalized space
    u_norm, v_norm = raw_to_normalized_pixels(u_raw, v_raw, norm_params, img_size)

    # 5. Apply roll on normalized pixels
    u_final, v_final = apply_roll_to_pixels(u_norm, v_norm, roll, img_size, img_size)

    # 6. Round & in-bounds
    u_pix = np.round(u_final).astype(np.int32)
    v_pix = np.round(v_final).astype(np.int32)

    H = W = img_size
    in_bounds = (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H)

    valid_mask = valid & in_bounds

    return pts, valid_mask, u_pix, v_pix

# ---------------- SEGMENTATION & OVERLAY ----------------

def parse_base_label(filename_stem: str):
    """
    Input: "backrest_1_mask"
    Output: "backrest"
    (for logging only)
    """
    if filename_stem.endswith("_mask"):
        base = filename_stem[:-5]
    else:
        return filename_stem

    parts = base.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0]
    return base

def load_component_masks(view_name):
    """
    Load all component masks for this view.

    Returns a list of tuples:
        [(component_id, seg_mask), ...]
    where component_id is mask_path.stem (e.g., 'backrest_0_mask') and
    seg_mask is a binary mask of shape (IMG_SIZE, IMG_SIZE).
    """
    current_seg_dir = SEG_DIR / view_name
    components = []

    if not current_seg_dir.exists():
        return components

    mask_files = sorted(list(current_seg_dir.glob("*_mask.png")))
    for mask_path in mask_files:
        component_id = mask_path.stem  # full stem, e.g., 'leg_0_mask'
        seg = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if seg is None:
            continue
        if seg.shape[:2] != (IMG_SIZE, IMG_SIZE):
            seg = cv2.resize(seg, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        seg_mask = (seg > 0).astype(np.uint8)
        components.append((component_id, seg_mask))

    return components

def apply_segmentation_overlays(base_render_bgr, components):
    """
    Colorize the base_render_bgr using component masks.

    Each component (mask file) gets its own color, even if labels repeat.
    """
    overlay_accum = base_render_bgr.copy()

    if not components:
        return base_render_bgr

    for component_id, seg_mask in components:
        color_bgr = get_color_for_component(component_id)
        overlay_accum[seg_mask > 0] = color_bgr
        # For logging, also show base label
        base_label = parse_base_label(component_id)
        print(f"   + Applied component {component_id} (base label: {base_label})")

    return overlay_accum

# ---------------- MAIN ----------------

def main():
    # Debug: print absolute path so user knows where we are looking
    print(f"--- 📂 Path Check ---")
    print(f"Script running in: {Path.cwd()}")
    print(f"Looking for PLY at: {PLY_PATH.resolve()}")
    print(f"Looking for JSON at: {JSON_PATH.resolve()}")
    print(f"Looking for SEG  at: {SEG_DIR.resolve()}")

    if not JSON_PATH.exists():
        print(f"❌ Error: JSON file not found!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, "r") as f:
        camera_data = json.load(f)

    # Setup Renderer
    geom = load_geometry(PLY_PATH)
    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    radius = 2.5 * float(np.max(bbox.get_extent()))
    if radius <= 0:
        radius = 1.0

    renderer = rendering.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.base_color = (0.0, 0.0, 0.0, 1.0)
    mat.point_size = 2.5
    renderer.scene.add_geometry("obj", geom, mat)

    center_pt = (IMG_SIZE // 2, IMG_SIZE // 2)

    print(f"\n--- 🎨 Processing Views ---")

    for view_name, params in camera_data.items():
        if "error" in params:
            continue

        print(f"\nRendering {view_name}...")

        az, el, roll = params['azimuth'], params['elevation'], params['roll']

        # 1. Base render: silhouette (raw) + normalization + roll
        raw_mask = render_solid_silhouette(renderer, center, radius, az, el)
        norm_mask, norm_params = normalize_mask(raw_mask, IMG_SIZE)

        M = cv2.getRotationMatrix2D(center_pt, roll, 1.0)
        final_render_mask = cv2.warpAffine(
            norm_mask, M, (IMG_SIZE, IMG_SIZE),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Create base image (Dark Grey object, White Background)
        base_img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
        base_img[final_render_mask > 0] = [60, 60, 60]

        # 2. Load component masks for this view
        components = load_component_masks(view_name)

        # 3. 2D Overlay (per-component colors)
        final_img = apply_segmentation_overlays(base_img, components)

        # Save 2D overlay
        out_path = OUTPUT_DIR / f"{view_name}_overlaid.png"
        cv2.imwrite(str(out_path), final_img)

        # 4. Precompute projection of all 3D points for this view
        pts, proj_valid, u_pix, v_pix = precompute_projection(
            geom, center, radius, az, el, roll, IMG_SIZE, norm_params
        )

        # 5. Initialize 3D colors: default gray everywhere
        N = len(pts)
        colors = np.zeros((N, 3), dtype=np.float32)
        colors[:] = np.array([0.6, 0.6, 0.6], dtype=np.float32)

        # 6. For each component, build a slightly dilated overlap mask,
        #    then color corresponding 3D points with that component's color.
        if components:
            kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)

            for component_id, seg_mask in components:
                # Dilate segmentation mask to be tolerant wrt misalignment
                seg_dilated = cv2.dilate(seg_mask.astype(np.uint8), kernel,
                                         iterations=DILATION_ITERATIONS)
                seg_dilated = (seg_dilated > 0)

                # Overlap in 2D (normalized + rolled) space:
                # only color where both dilated seg and rendered silhouette exist
                overlap_2d = seg_dilated & (final_render_mask > 0)

                # Now push that assignment to 3D
                idx = np.where(proj_valid)[0]
                if idx.size > 0:
                    # For valid points, check overlap_2d at their pixel coordinates
                    ov = overlap_2d[v_pix[idx], u_pix[idx]]

                    # Choose color for this component (convert BGR -> float RGB-ish)
                    c_bgr = np.array(get_color_for_component(component_id), dtype=np.float32)
                    c_rgb = c_bgr[::-1] / 255.0  # reverse to approx RGB, then normalize

                    colors[idx[ov]] = c_rgb

        # 7. Save colored 3D geometry for this view
        if isinstance(geom, o3d.geometry.PointCloud):
            geom_colored = o3d.geometry.PointCloud(geom)
            geom_colored.colors = o3d.utility.Vector3dVector(colors)
            out_ply = OUTPUT_DIR / f"{view_name}_overlap3d.ply"
            o3d.io.write_point_cloud(str(out_ply), geom_colored)
        else:
            geom_colored = o3d.geometry.TriangleMesh(geom)
            geom_colored.vertex_colors = o3d.utility.Vector3dVector(colors)
            out_ply = OUTPUT_DIR / f"{view_name}_overlap3d_mesh.ply"
            o3d.io.write_triangle_mesh(str(out_ply), geom_colored)

        print(f"   3D overlap saved to: {out_ply.name}")

    print(f"\n✅ Success! 2D images + 3D overlaps saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
