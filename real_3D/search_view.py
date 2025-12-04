#!/usr/bin/env python3
"""
camera_search_parallel.py

Two-stage pose search with parallel coarse phase:

1) Coarse search (PARALLEL over base rotations):
   - 16 base rotations (rx, ry in {0,90,180,270}°)
   - per base: az, el grid with 10° steps

2) Fine search (SINGLE PROCESS):
   - around best coarse (base, az, el)
   - az, el in [best-10, best+10] with 1° step

Outputs in pose_best_view/:
  - shape.png      : rendered shape only at best pose
  - overlay.png    : sketch vs best silhouette overlay
  - camera.json    : base rotation, azimuth, elevation, radius, center, score
"""

import os
os.environ["OPEN3D_LOG_LEVEL"] = "error"  # suppress [Open3D INFO]
os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"  # avoid EGL noise
os.environ["FILAMENT_DISABLE_COMPUTE"] = "1"  # reduce backend spam


from pathlib import Path
import math
import itertools
import copy
import json
import multiprocessing as mp

import numpy as np
import cv2
import open3d as o3d
from open3d.visualization import rendering


# ---------------- paths & config ----------------

PLY_PATH    = Path("trellis_outputs/0_trellis_gaussian.ply")
SKETCH_PATH = Path("0.png")

OUT_DIR = Path("pose_best_view")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHAPE_IMG   = OUT_DIR / "shape.png"
OVERLAY_IMG = OUT_DIR / "overlay.png"
CAMERA_JSON = OUT_DIR / "camera.json"

# Render size (small & fast)
IMG_SIZE = 160

# Coarse search grid
COARSE_AZ_STEP = 15.0
COARSE_EL_STEP = 15.0
AZ_MIN, AZ_MAX = -180.0, 180.0
EL_MIN, EL_MAX = -60.0, 60.0

# Fine search window and step
FINE_AZ_DELTA = 15.0
FINE_EL_DELTA = 15.0
FINE_AZ_STEP  = 0.5
FINE_EL_STEP  = 0.5

# Parallel workers for coarse search
# (Each worker uses its own OffscreenRenderer + PLY load)
NUM_WORKERS = min(12, mp.cpu_count() // 2)


# ---------------- helpers ----------------

def load_geometry(path: Path):
    """Load PLY as point cloud or mesh."""
    geom = o3d.io.read_point_cloud(str(path))
    if geom.is_empty():
        geom = o3d.io.read_triangle_mesh(str(path))
        if geom.is_empty():
            raise RuntimeError("PLY is empty (neither point cloud nor mesh).")
        geom.compute_vertex_normals()
    return geom


def make_material(geom):
    """Simple dark material for point cloud or mesh."""
    mat = rendering.MaterialRecord()
    if isinstance(geom, o3d.geometry.PointCloud):
        mat.shader = "defaultUnlit"
        mat.point_size = 1.5
    else:
        mat.shader = "defaultLit"
    mat.base_color = (0.1, 0.1, 0.1, 1.0)  # dark grey
    return mat


def load_sketch_silhouette(path: Path, canvas_size: int) -> np.ndarray:
    """
    Load sketch, convert to a filled silhouette mask, then normalize.
    Assumes white background, darker lines.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read sketch: {path}")

    img = cv2.resize(img, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)

    # Make lines bright on dark for easier thresholding
    if np.mean(img) > 127:
        img = 255 - img

    # Threshold: anything with ink becomes foreground
    _, mask = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # Fill closed external contours to form solid silhouette
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(mask)
    cv2.drawContours(sil, contours, -1, 255, thickness=-1)

    sil = (sil > 0).astype(np.uint8)

    norm = normalize_mask(sil, canvas_size=canvas_size)
    return norm


def normalize_mask(mask: np.ndarray, canvas_size: int, margin: float = 0.1) -> np.ndarray:
    """
    Align a binary mask to a centered, normalized canvas.
    """
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    crop = mask[y_min:y_max + 1, x_min:x_max + 1]
    ch, cw = crop.shape

    target_side = int((1.0 - 2.0 * margin) * canvas_size)
    target_side = max(1, target_side)

    scale = target_side / max(ch, cw)
    new_h = max(1, int(round(ch * scale)))
    new_w = max(1, int(round(cw * scale)))

    resized = cv2.resize(crop.astype(np.uint8) * 255,
                         (new_w, new_h),
                         interpolation=cv2.INTER_AREA)
    resized = (resized > 0).astype(np.uint8)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    start_y = (canvas_size - new_h) // 2
    start_x = (canvas_size - new_w) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    return canvas


def sph_dir(az_deg, el_deg):
    """Spherical direction: +Y forward, +X right, +Z up."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = math.cos(el) * math.sin(az)
    y = math.cos(el) * math.cos(az)
    z = math.sin(el)
    return np.array([x, y, z], dtype=float)


def silhouette_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Combined metric:
      - IoU on silhouettes
      - Area ratio penalty for size mismatch
    """
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0

    iou = inter / union

    area_a = a.sum()
    area_b = b.sum()
    if area_a == 0 or area_b == 0:
        area_ratio = 0.0
    else:
        area_ratio = min(area_a, area_b) / max(area_a, area_b)

    score = 0.7 * iou + 0.3 * area_ratio
    return float(score)


def generate_base_rotations():
    """
    Generate 16 90°-step rotations for the OBJECT:
    rx, ry ∈ {0, 90, 180, 270} degrees, deduplicated.
    """
    angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    all_rots = []
    for rx, ry in itertools.product(angles, angles):
        R = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, 0.0))
        all_rots.append(R)

    unique = []
    for R in all_rots:
        if not any(np.allclose(R, U, atol=1e-6) for U in unique):
            unique.append(R)
    return unique


def set_camera(renderer, center, radius, az_deg, el_deg):
    """Configure camera on renderer for given spherical angles."""
    scene = renderer.scene
    direction = sph_dir(az_deg, el_deg)
    eye = center + radius * direction
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    cam = scene.camera
    cam.set_projection(40.0, 1.0, radius * 0.1, radius * 10.0,
                       rendering.Camera.FovType.Vertical)
    cam.look_at(center, eye, up)


def render_silhouette(renderer, center, radius, az_deg, el_deg) -> np.ndarray:
    """Render scene, then return filled silhouette mask."""
    set_camera(renderer, center, radius, az_deg, el_deg)
    img_o3d = renderer.render_to_image()
    img = np.asarray(img_o3d)  # H x W x 3/4

    # Convert to grayscale
    if img.shape[2] == 4:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(img_gray, 50, 150)

    # Fill external contours to get silhouette
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(edges)
    cv2.drawContours(sil, contours, -1, 255, thickness=-1)

    sil = (sil > 0).astype(np.uint8)
    return sil


def render_rgb_and_silhouette(renderer, center, radius, az_deg, el_deg):
    """Render RGB + silhouette (for final best pose)."""
    set_camera(renderer, center, radius, az_deg, el_deg)
    img_o3d = renderer.render_to_image()
    img = np.asarray(img_o3d)  # H x W x 3/4

    # Convert to grayscale + keep RGB
    if img.shape[2] == 4:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb = img.copy()

    # Edge detection
    edges = cv2.Canny(img_gray, 50, 150)

    # Fill external contours to get silhouette
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(edges)
    cv2.drawContours(sil, contours, -1, 255, thickness=-1)

    sil = (sil > 0).astype(np.uint8)
    return rgb, sil


def create_overlay(norm_sketch: np.ndarray,
                   norm_render: np.ndarray,
                   out_path: Path):
    """
    Create a single overlay image:
      - white background
      - grey  where render silhouette
      - red   where sketch silhouette (alpha blend)
    """
    H, W = norm_sketch.shape
    overlay = np.ones((H, W, 3), dtype=np.uint8) * 255

    grey = np.array([180, 180, 180], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)
    alpha = 0.7

    r_idx = norm_render > 0
    s_idx = norm_sketch > 0

    overlay[r_idx] = grey
    overlay[s_idx] = (alpha * red + (1 - alpha) * overlay[s_idx]).astype(np.uint8)

    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# ---------------- coarse worker (parallel) ----------------

def coarse_worker(args):
    """
    Worker for coarse search on a single base rotation.

    Args:
        args: (base_idx, sketch_sil_norm)
    Returns:
        (best_score, base_idx, best_az, best_el)
    """
    base_idx, sketch_sil_norm = args

    # Generate same base rotations locally (small & deterministic)
    base_rots = generate_base_rotations()
    R = base_rots[base_idx]

    geom_orig = load_geometry(PLY_PATH)

    # Rotate geometry by this base
    geom = copy.deepcopy(geom_orig)
    bbox0 = geom.get_axis_aligned_bounding_box()
    center0 = bbox0.get_center()
    geom.rotate(R, center=center0)

    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = 2.5 * float(np.max(extent))

    # Local renderer for this process
    renderer = rendering.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    mat = make_material(geom)
    scene = renderer.scene
    scene.clear_geometry()
    scene.add_geometry(f"obj_base_{base_idx}", geom, mat)

    azimuths = np.arange(AZ_MIN, AZ_MAX + 1e-6, COARSE_AZ_STEP)
    elevations = np.arange(EL_MIN, EL_MAX + 1e-6, COARSE_EL_STEP)
    total = len(azimuths) * len(elevations)

    best_score = -1.0
    best_az = None
    best_el = None

    print(f"[Worker base {base_idx}] starting coarse search over {total} views...")

    idx = 0
    for el in elevations:
        for az in azimuths:
            idx += 1
            sil_r_raw = render_silhouette(renderer, center, radius, az, el)
            render_norm = normalize_mask(sil_r_raw, canvas_size=IMG_SIZE)

            score = silhouette_score(sketch_sil_norm, render_norm)

            if score > best_score:
                best_score = score
                best_az = az
                best_el = el

            if idx % 100 == 0:
                print(f"[Worker base {base_idx}] "
                      f"{idx:4d}/{total} | local best={best_score:.4f}")

    print(f"[Worker base {base_idx}] done. best_score={best_score:.4f}, "
          f"az={best_az:.1f}, el={best_el:.1f}")

    # Let process clean up its renderer when it exits
    return best_score, base_idx, best_az, best_el


# ---------------- main ----------------

def main():
    print("\n⏱️ Starting CAMERA SEARCH (parallel coarse, single fine)")

    # Load sketch silhouette once in parent
    sketch_sil_norm = load_sketch_silhouette(SKETCH_PATH, canvas_size=IMG_SIZE)

    # Generate base rotations (16)
    base_rots = generate_base_rotations()
    num_bases = len(base_rots)
    print(f"Using all {num_bases} base rotations: {list(range(num_bases))}")

    # -------- parallel coarse search over bases --------
    print("\n----- 🟦 COARSE SEARCH (parallel over bases) -----")
    args_list = [(base_idx, sketch_sil_norm) for base_idx in range(num_bases)]

    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(coarse_worker, args_list)

    # results: list of (best_score, base_idx, best_az, best_el)
    global_best_score = -1.0
    global_best_base_idx = None
    global_best_az = None
    global_best_el = None

    for best_score, base_idx, best_az, best_el in results:
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_base_idx = base_idx
            global_best_az = best_az
            global_best_el = best_el

    print(f"\n🌟 Best coarse result:"
          f"\n  base_rotation_index = {global_best_base_idx}"
          f"\n  az = {global_best_az:.1f}, el = {global_best_el:.1f}"
          f"\n  score = {global_best_score:.4f}")

    # -------- fine search for the best base (single process) --------
    print("\n----- 🟩 FINE SEARCH (single process) -----")

    geom_orig = load_geometry(PLY_PATH)
    best_R = base_rots[global_best_base_idx]

    geom = copy.deepcopy(geom_orig)
    bbox0 = geom.get_axis_aligned_bounding_box()
    center0 = bbox0.get_center()
    geom.rotate(best_R, center=center0)

    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = 2.5 * float(np.max(extent))

    renderer = rendering.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    scene = renderer.scene
    scene.clear_geometry()
    scene.add_geometry("obj_best_base", geom, make_material(geom))

    azimuths_fine = np.arange(global_best_az - FINE_AZ_DELTA,
                              global_best_az + FINE_AZ_DELTA + 1e-6,
                              FINE_AZ_STEP)
    elevations_fine = np.arange(global_best_el - FINE_EL_DELTA,
                                global_best_el + FINE_EL_DELTA + 1e-6,
                                FINE_EL_STEP)
    total_fine = len(azimuths_fine) * len(elevations_fine)

    fine_best_score = -1.0
    fine_best_az = None
    fine_best_el = None
    fine_best_render_norm = None

    idx = 0
    for el in elevations_fine:
        for az in azimuths_fine:
            idx += 1
            sil_r_raw = render_silhouette(renderer, center, radius, az, el)
            render_norm = normalize_mask(sil_r_raw, canvas_size=IMG_SIZE)

            score = silhouette_score(sketch_sil_norm, render_norm)

            if score > fine_best_score:
                fine_best_score = score
                fine_best_az = az
                fine_best_el = el
                fine_best_render_norm = render_norm.copy()
                print(f"  New fine best: score={fine_best_score:.4f} | "
                      f"az={fine_best_az:.1f}, el={fine_best_el:.1f}")

            if idx % 200 == 0:
                print(f"  [fine {idx:4d}/{total_fine}] current_best={fine_best_score:.4f}")

    print("\n🎯 FINAL BEST:"
          f"\n  az = {fine_best_az:.1f}, el = {fine_best_el:.1f}, "
          f"score = {fine_best_score:.4f}")

    # -------- final render + outputs --------
    best_rgb, best_sil_raw = render_rgb_and_silhouette(
        renderer, center, radius, fine_best_az, fine_best_el
    )
    best_rgb_resized = cv2.resize(best_rgb, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(SHAPE_IMG), best_rgb_resized)

    # Use fine_best_render_norm for overlay (already normalized)
    create_overlay(sketch_sil_norm, fine_best_render_norm, OVERLAY_IMG)

    camera_info = {
        "base_rotation": best_R.tolist(),  # 3x3 matrix
        "azimuth_deg": float(fine_best_az),
        "elevation_deg": float(fine_best_el),
        "radius": float(radius),
        "center": [float(center[0]), float(center[1]), float(center[2])],
        "score": float(fine_best_score),
    }
    with CAMERA_JSON.open("w") as f:
        json.dump(camera_info, f, indent=2)

    print(f"\n📁 Results saved to: {OUT_DIR}")
    print("   - shape.png")
    print("   - overlay.png")
    print("   - camera.json")
    print("\n✨ Done!")


if __name__ == "__main__":
    # On Linux, spawn is safer with GPU/Open3D than fork
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # Start method was already set
        pass
    main()
