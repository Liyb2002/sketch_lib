#!/usr/bin/env python3
"""
render_six_views.py

Goal:
  Take the aligned camera from pose_best_view/camera.json
  and render the same PLY shape from the 6 Zero123++ novel
  view poses (relative azimuths, fixed absolute elevations),
  AND create overlays between each rendered view and the
  corresponding sketch view_{i}.png in a /views folder.

Inputs:
  - PLY_PATH    : your reconstructed shape
  - CAMERA_JSON : pose_best_view/camera.json from the search script
  - SKETCH_VIEWS_DIR: folder containing view_0.png, view_1.png, ...

Outputs (in the SAME folder as CAMERA_JSON):
  - ply_view_0.png ... ply_view_5.png
  - overlay_view_0.png ... overlay_view_5.png
  - ply_six_views_camera.json
"""

from pathlib import Path
import json
import math

import numpy as np
import cv2
import open3d as o3d
from open3d.visualization import rendering


# ---------------- config ----------------

# Your shape
PLY_PATH = Path("trellis_outputs/0_trellis_gaussian.ply")

# Camera from the previous search
CAMERA_JSON = Path("pose_best_view/camera.json")

# Folder that already contains view_0.png, view_1.png, ...
# (these are the sketch / Zero123++ views you want to compare against)
SKETCH_VIEWS_DIR = Path("views")

# Render size
IMG_SIZE = 128  # 128 x 128, as discussed


# ---------------- helpers ----------------

def load_geometry(path: Path):
    """Load PLY as point cloud or mesh."""
    geom = o3d.io.read_point_cloud(str(path))
    if geom.is_empty():
        geom = o3d.io.read_triangle_mesh(str(path))
        if geom.is_empty():
            raise RuntimeError(f"PLY is empty at: {path}")
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


def sph_dir(az_deg: float, el_deg: float):
    """
    Spherical direction in our convention:
      +Y forward, +X right, +Z up
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = math.cos(el) * math.sin(az)
    y = math.cos(el) * math.cos(az)
    z = math.sin(el)
    return np.array([x, y, z], dtype=float)


def wrap_deg_signed(angle_deg: float) -> float:
    """
    Wrap angle to [-180, 180).
    """
    return (angle_deg + 180.0) % 360.0 - 180.0


def set_camera(renderer, center, radius, az_deg, el_deg):
    """Configure camera on renderer for given spherical angles."""
    scene = renderer.scene
    direction = sph_dir(az_deg, el_deg)
    eye = center + radius * direction
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    cam = scene.camera
    cam.set_projection(
        40.0,  # vertical FOV in degrees
        1.0,   # aspect ratio (square)
        radius * 0.1,
        radius * 10.0,
        rendering.Camera.FovType.Vertical,
    )
    cam.look_at(center, eye, up)
    return eye


def rgb_to_silhouette(img_bgr: np.ndarray) -> np.ndarray:
    """
    Given a BGR image, compute a filled silhouette:
      - grayscale
      - Canny edges
      - fill external contours
    Returns a binary mask (uint8 0/1).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(gray)
    cv2.drawContours(sil, contours, -1, 255, thickness=-1)
    return (sil > 0).astype(np.uint8)


def create_overlay(sil_sketch: np.ndarray,
                   sil_ply: np.ndarray,
                   out_path: Path):
    """
    Create a single overlay image:
      - white background
      - grey  where PLY silhouette
      - red   where sketch silhouette (alpha blend)
    """
    H, W = sil_sketch.shape
    overlay = np.ones((H, W, 3), dtype=np.uint8) * 255

    grey = np.array([180, 180, 180], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)
    alpha = 0.7

    r_idx = sil_ply > 0
    s_idx = sil_sketch > 0

    overlay[r_idx] = grey
    overlay[s_idx] = (alpha * red + (1 - alpha) * overlay[s_idx]).astype(np.uint8)

    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# ---------------- main ----------------

def main():
    if not CAMERA_JSON.is_file():
        raise FileNotFoundError(f"Camera JSON not found: {CAMERA_JSON}")
    if not PLY_PATH.is_file():
        raise FileNotFoundError(f"PLY not found: {PLY_PATH}")
    if not SKETCH_VIEWS_DIR.is_dir():
        raise FileNotFoundError(f"Sketch views folder not found: {SKETCH_VIEWS_DIR}")

    # Save everything next to camera.json
    out_dir = CAMERA_JSON.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load base camera info
    with CAMERA_JSON.open("r") as f:
        cam = json.load(f)

    base_R = np.array(cam["base_rotation"], dtype=float)  # 3x3
    az_in = float(cam["azimuth_deg"])
    el_in = float(cam["elevation_deg"])  # kept for record

    # 2) Load and rotate geometry by base_R
    geom = load_geometry(PLY_PATH)

    bbox0 = geom.get_axis_aligned_bounding_box()
    center0 = bbox0.get_center()
    geom.rotate(base_R, center=center0)

    # Recompute bbox after rotation
    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = 2.5 * float(np.max(extent))

    # 3) Set up renderer
    renderer = rendering.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white

    scene = renderer.scene
    scene.clear_geometry()
    scene.add_geometry("obj", geom, make_material(geom))

    # 4) Zero123++ 6-view pattern
    rel_azimuths = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]
    abs_elevations = [20.0, -10.0, 20.0, -10.0, 20.0, -10.0]

    views_info = []

    for idx, (rel_az, abs_el) in enumerate(zip(rel_azimuths, abs_elevations)):
        az = wrap_deg_signed(az_in + rel_az)
        el = abs_el  # Zero123++ uses fixed absolute elevations

        # Set camera and render PLY
        eye = set_camera(renderer, center, radius, az, el)
        img_o3d = renderer.render_to_image()
        img = np.asarray(img_o3d)  # H x W x 3/4

        # Convert BGRA -> BGR if needed
        if img.shape[2] == 4:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img_bgr = img.copy()

        # Save PLY render
        ply_view_name = f"ply_view_{idx}.png"
        ply_view_path = out_dir / ply_view_name
        cv2.imwrite(str(ply_view_path), img_bgr)

        # Load corresponding sketch view_{idx}.png from SKETCH_VIEWS_DIR
        sketch_view_path = SKETCH_VIEWS_DIR / f"view_{idx}.png"
        if sketch_view_path.is_file():
            sketch_bgr = cv2.imread(str(sketch_view_path), cv2.IMREAD_COLOR)
            # Resize sketch to match IMG_SIZE (just in case)
            sketch_bgr_resized = cv2.resize(sketch_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            # Compute silhouettes
            sil_sketch = rgb_to_silhouette(sketch_bgr_resized)
            sil_ply = rgb_to_silhouette(img_bgr)

            # Create overlay
            overlay_path = out_dir / f"overlay_view_{idx}.png"
            create_overlay(sil_sketch, sil_ply, overlay_path)
        else:
            # If the sketch view is missing, we just skip overlay
            # (no noisy prints unless it's actually missing)
            pass

        views_info.append(
            {
                "index": idx,
                "ply_image": ply_view_name,
                "relative_azimuth_deg": rel_az,
                "azimuth_deg": float(az),
                "elevation_deg": float(el),
                "radius": float(radius),
                "center": [float(center[0]), float(center[1]), float(center[2])],
                "eye": [float(eye[0]), float(eye[1]), float(eye[2])],
            }
        )

    # 5) Save a small JSON describing these 6 cameras
    out_json = {
        "note": (
            "Six PLY render views corresponding to Zero123++ novel-view cameras. "
            "Azimuths = input_azimuth + [30, 90, 150, 210, 270, 330] (wrapped to [-180,180)). "
            "Elevations = [20, -10, 20, -10, 20, -10] degrees. "
            "Overlays (overlay_view_i.png) compare silhouettes of sketch views/view_i.png "
            "and rendered ply_view_i.png."
        ),
        "input_camera": {
            "azimuth_deg": az_in,
            "elevation_deg": el_in,
            "base_rotation": cam["base_rotation"],
        },
        "views": views_info,
    }

    json_path = out_dir / "ply_six_views_camera.json"
    with json_path.open("w") as f:
        json.dump(out_json, f, indent=2)

    print(f"Saved 6 PLY views + overlays + camera JSON to: {out_dir}")


if __name__ == "__main__":
    main()
