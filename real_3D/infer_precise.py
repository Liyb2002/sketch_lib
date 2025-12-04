#!/usr/bin/env python3
"""
fine_search_view.py

Fine search around a known good pose:

    base_rotation_index = 15
    az_center = 20.0
    el_center = 30.0

We search:
    azimuth  in [az_center - 10, az_center + 10] with step 2
    elevation in [el_center - 10, el_center + 10] with step 2

using the same silhouette-based metric as before.

For each candidate pose:
    - render the object
    - extract silhouette from RGB (edges + external contour fill)
    - normalize both sketch and render silhouettes
    - compute score = 0.7 * IoU + 0.3 * area_ratio
    - save overlay to pose_search_fine/{score:.4f}.png

After the best pose is found:
    - render ONLY the shape (no overlay) at that pose
    - save to best_shape_view.png
"""

from pathlib import Path
import math
import itertools
import copy

import numpy as np
import cv2
import open3d as o3d
from open3d.visualization import rendering


# ---------------- paths & config ----------------

PLY_PATH    = Path("trellis_outputs/0_trellis_gaussian.ply")
SKETCH_PATH = Path("0.png")  # your sketch filename

OUT_FINE    = Path("pose_search_fine")
OUT_FINE.mkdir(parents=True, exist_ok=True)

BEST_SHAPE_IMG = Path("best_shape_view.png")

IMG_SIZE = 320  # output canvas size (square)

# from your coarse search result:
BASE_ROT_INDEX = 15
AZ_CENTER = 20.0
EL_CENTER = 30.0

AZ_DELTA = 10.0
EL_DELTA = 10.0
AZ_STEP  = 2.0
EL_STEP  = 2.0


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


def load_sketch_silhouette(path: Path) -> np.ndarray:
    """
    Load 0.png, convert to a filled silhouette mask.
    Assumes white background, darker lines.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read sketch: {path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

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
    return sil


def normalize_mask(mask: np.ndarray, canvas_size: int = IMG_SIZE,
                   margin: float = 0.1) -> np.ndarray:
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


def save_overlay(norm_sketch: np.ndarray,
                 norm_render: np.ndarray,
                 score: float):
    """
    Save an overlay image named '{score:.4f}.png' in OUT_FINE:
    - white background
    - grey  where render silhouette
    - red   where sketch silhouette (alpha blend)
    Also prints a line once each image is saved.
    """
    H, W = norm_sketch.shape
    overlay = np.ones((H, W, 3), dtype=np.uint8) * 255

    grey = np.array([180, 180, 180], dtype=np.uint8)
    red  = np.array([255, 0, 0], dtype=np.uint8)
    alpha = 0.7

    r_idx = norm_render > 0
    s_idx = norm_sketch > 0

    overlay[r_idx] = grey
    overlay[s_idx] = (alpha * red + (1 - alpha) * overlay[s_idx]).astype(np.uint8)

    filename = f"{score:.4f}.png"
    out_path = OUT_FINE / filename
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Saved {filename}")


def generate_base_rotations():
    """
    Same as in coarse search:
    Rotations around X and Y by {0, 90, 180, 270} degrees, deduplicated.
    """
    angles = [0, math.pi/2, math.pi, 3*math.pi/2]
    all_rots = []
    for rx, ry in itertools.product(angles, angles):
        R = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, 0.0))
        all_rots.append(R)

    unique = []
    for R in all_rots:
        if not any(np.allclose(R, U, atol=1e-6) for U in unique):
            unique.append(R)
    return unique


def render_rgb_and_silhouette(renderer, center, radius, az_deg, el_deg):
    """
    Render RGB using current scene geometry, and extract silhouette from RGB:
        - grayscale -> Canny
        - external contours -> fill
    Returns (rgb_img, sil_mask)
    """
    scene = renderer.scene
    direction = sph_dir(az_deg, el_deg)
    eye = center + radius * direction
    up  = np.array([0.0, 0.0, 1.0], dtype=float)

    cam = scene.camera
    cam.set_projection(40.0, 1.0, radius * 0.1, radius * 10.0,
                       rendering.Camera.FovType.Vertical)
    cam.look_at(center, eye, up)

    img_o3d = renderer.render_to_image()
    img = np.asarray(img_o3d)  # H x W x 3/4

    # Convert to grayscale
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


# ---------------- main ----------------

def main():
    geom_orig = load_geometry(PLY_PATH)
    sketch_sil_raw  = load_sketch_silhouette(SKETCH_PATH)
    sketch_sil_norm = normalize_mask(sketch_sil_raw)

    # get the same base rotations list and pick the one at BASE_ROT_INDEX
    base_rots = generate_base_rotations()
    if BASE_ROT_INDEX < 0 or BASE_ROT_INDEX >= len(base_rots):
        raise ValueError(f"BASE_ROT_INDEX {BASE_ROT_INDEX} out of range (0..{len(base_rots)-1})")
    R = base_rots[BASE_ROT_INDEX]

    print(f"Fine search with base_rotation_index = {BASE_ROT_INDEX}")

    # prepare geometry with this base rotation
    geom = copy.deepcopy(geom_orig)
    bbox0 = geom.get_axis_aligned_bounding_box()
    center0 = bbox0.get_center()
    geom.rotate(R, center=center0)

    # recompute bbox & radius after rotation
    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    max_extent = float(np.max(extent))
    radius = 2.5 * max_extent

    mat = make_material(geom)

    renderer = rendering.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    scene = renderer.scene
    scene.clear_geometry()
    scene.add_geometry("obj", geom, mat)

    # define fine search grid
    azimuths = np.arange(AZ_CENTER - AZ_DELTA, AZ_CENTER + AZ_DELTA + 1e-6, AZ_STEP)
    elevations = np.arange(EL_CENTER - EL_DELTA, EL_CENTER + EL_DELTA + 1e-6, EL_STEP)

    total = len(azimuths) * len(elevations)
    idx = 0

    best_score = -1.0
    best_az = None
    best_el = None
    best_rgb = None

    for el in elevations:
        for az in azimuths:
            idx += 1

            rgb, sil_r_raw = render_rgb_and_silhouette(renderer, center, radius, az, el)
            render_norm = normalize_mask(sil_r_raw)

            score = silhouette_score(sketch_sil_norm, render_norm)
            print(f"[{idx:4d}/{total}] az={az:6.1f}, el={el:6.1f}, score={score:.4f}")

            save_overlay(sketch_sil_norm, render_norm, score)

            if score > best_score:
                best_score = score
                best_az = az
                best_el = el
                best_rgb = rgb.copy()

    print(f"\nFINE GLOBAL BEST: score={best_score:.4f}, az={best_az:.1f}, el={best_el:.1f}")

    # Save ONLY the shape rendered at the best pose
    if best_rgb is not None:
        # optionally resize to IMG_SIZE x IMG_SIZE for consistency
        best_rgb_resized = cv2.resize(best_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(BEST_SHAPE_IMG), best_rgb_resized)
        print(f"Saved best shape-only view to: {BEST_SHAPE_IMG}")
    else:
        print("No best_rgb found, something went wrong.")


if __name__ == "__main__":
    main()
