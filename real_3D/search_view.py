#!/usr/bin/env python3
"""
camera_search_coarse_fine.py  (with good logging)

Two-stage pose search:
  1) Coarse: explore base rotations + 10° grid
  2) Fine: 1° sweep around the best coarse pose

Outputs in pose_best_view/:
   shape.png, overlay.png, camera.json
"""

from pathlib import Path
import math
import itertools
import copy
import json

import numpy as np
import cv2
import open3d as o3d
from open3d.visualization import rendering


# ---------------- config ----------------
PLY_PATH    = Path("trellis_outputs/0_trellis_gaussian.ply")
SKETCH_PATH = Path("0.png")

OUT_DIR = Path("pose_best_view")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHAPE_IMG   = OUT_DIR / "shape.png"
OVERLAY_IMG = OUT_DIR / "overlay.png"
CAMERA_JSON = OUT_DIR / "camera.json"

IMG_SIZE = 128  # small + fast

COARSE_AZ_STEP = 15.0
COARSE_EL_STEP = 15.0
AZ_MIN, AZ_MAX = -180.0, 180.0
EL_MIN, EL_MAX = -60.0, 60.0

FINE_AZ_DELTA = 15.0
FINE_EL_DELTA = 15.0
FINE_AZ_STEP  = 0.5
FINE_EL_STEP  = 0.5


# ---------------- helpers (unchanged) ----------------
def load_geometry(path: Path):
    geom = o3d.io.read_point_cloud(str(path))
    if geom.is_empty():
        geom = o3d.io.read_triangle_mesh(str(path))
        geom.compute_vertex_normals()
    return geom


def make_material(geom):
    mat = rendering.MaterialRecord()
    if isinstance(geom, o3d.geometry.PointCloud):
        mat.shader = "defaultUnlit"
        mat.point_size = 1.5
    else:
        mat.shader = "defaultLit"
    mat.base_color = (0.1, 0.1, 0.1, 1.0)
    return mat


def load_sketch_silhouette(path: Path, canvas_size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (canvas_size, canvas_size))
    if np.mean(img) > 127:
        img = 255 - img
    _, mask = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(mask)
    cv2.drawContours(sil, contours, -1, 255, -1)
    sil = (sil > 0).astype(np.uint8)
    return normalize_mask(sil, canvas_size)


def normalize_mask(mask, canvas_size, margin=0.1):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((canvas_size, canvas_size), np.uint8)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    crop = mask[y_min:y_max+1, x_min:x_max+1]
    ch, cw = crop.shape
    scale = int((1 - 2*margin) * canvas_size) / max(ch, cw)
    new_h, new_w = max(1, int(ch * scale)), max(1, int(cw * scale))
    resized = cv2.resize(crop.astype(np.uint8)*255, (new_w,new_h)) > 0
    canvas = np.zeros((canvas_size, canvas_size), np.uint8)
    sy, sx = (canvas_size-new_h)//2, (canvas_size-new_w)//2
    canvas[sy:sy+new_h, sx:sx+new_w] = resized.astype(np.uint8)
    return canvas


def sph_dir(az, el):
    az, el = map(math.radians, [az, el])
    return np.array([math.cos(el)*math.sin(az),
                     math.cos(el)*math.cos(az),
                     math.sin(el)], float)


def silhouette_score(a, b):
    union = np.logical_or(a,b).sum()
    if union == 0: return 0.0
    inter = np.logical_and(a,b).sum()
    iou = inter/union
    ar = min(a.sum(),b.sum())/max(a.sum(),b.sum()) if a.sum()*b.sum()>0 else 0.0
    return 0.7*iou + 0.3*ar


def generate_base_rotations():
    ang = [0, math.pi/2, math.pi, 3*math.pi/2]
    allR = [o3d.geometry.get_rotation_matrix_from_xyz((rx,ry,0))
            for rx,ry in itertools.product(ang,ang)]
    uniq=[]
    for r in allR:
        if not any(np.allclose(r,u,1e-6) for u in uniq): uniq.append(r)
    return uniq


def set_cam(renderer, center, radius, az, el):
    eye = center + radius*sph_dir(az,el)
    renderer.scene.camera.set_projection(40.,1., radius*0.1, radius*10.,
                                         rendering.Camera.FovType.Vertical)
    renderer.scene.camera.look_at(center, eye, (0,0,1))


def render_silhouette(renderer, center, radius, az, el):
    set_cam(renderer,center,radius,az,el)
    img = np.asarray(renderer.render_to_image())
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) if img.shape[2]==4 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150)
    cnt,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    sil=np.zeros_like(edges); cv2.drawContours(sil,cnt,-1,255,-1)
    return (sil>0).astype(np.uint8)


def render_rgb_and_sil(renderer, center, radius, az, el):
    set_cam(renderer, center, radius, az, el)
    img = np.asarray(renderer.render_to_image())
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) if img.shape[2]==4 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150)
    cnt,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    sil=np.zeros_like(edges); cv2.drawContours(sil,cnt,-1,255,-1)
    return img.copy(), (sil>0).astype(np.uint8)


def save_overlay(sketch, render, dst):
    h,w = sketch.shape
    ov = np.ones((h,w,3),np.uint8)*255
    ov[render>0] = (180,180,180)
    s = sketch>0
    ov[s] = (0.7*np.array([255,0,0]) + 0.3*ov[s]).astype(np.uint8)
    cv2.imwrite(str(dst),cv2.cvtColor(ov,cv2.COLOR_RGB2BGR))


# ---------------- main ----------------
def main():
    print("\n⏱️ Starting CAMERA SEARCH")
    geom_orig = load_geometry(PLY_PATH)
    sketch = load_sketch_silhouette(SKETCH_PATH, IMG_SIZE)
    bases = generate_base_rotations()

    renderer = rendering.OffscreenRenderer(IMG_SIZE,IMG_SIZE)
    renderer.scene.set_background([1,1,1,1])

    global_best = (-1,None,None,None) # score, base_idx, az, el

    azs = np.arange(AZ_MIN,AZ_MAX+1e-6,COARSE_AZ_STEP)
    els = np.arange(EL_MIN,EL_MAX+1e-6,COARSE_EL_STEP)

    print("\n----- 🟦 COARSE SEARCH -----")

    for bi,R in enumerate(bases):
        geom = copy.deepcopy(geom_orig)
        cn = geom.get_axis_aligned_bounding_box().get_center()
        geom.rotate(R,cn)
        bbox=geom.get_axis_aligned_bounding_box()
        center=bbox.get_center(); radius=2.5*max(bbox.get_extent())

        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("obj",geom,make_material(geom))

        best_local_score=-1
        for el in els:
            for az in azs:
                s = silhouette_score(sketch, normalize_mask(render_silhouette(renderer,center,radius,az,el), IMG_SIZE))
                if s>best_local_score: best_local_score=s
                if s>global_best[0]:
                    global_best=(s,bi,az,el)
        print(f"  Base {bi+1:02d}/{len(bases)} | best local score = {best_local_score:.4f} | GLOBAL = {global_best[0]:.4f}")

    best_score, best_bi, baz, bel = global_best
    print(f"\n🌟 Best coarse result so far:"
          f"\n  Base index = {best_bi}"
          f"\n  az = {baz:.1f}, el = {bel:.1f}"
          f"\n  score = {best_score:.4f}")

    print("\n----- 🟩 FINE SEARCH -----")
    R = bases[best_bi]
    geom = copy.deepcopy(geom_orig)
    cn = geom.get_axis_aligned_bounding_box().get_center()
    geom.rotate(R,cn)
    bbox=geom.get_axis_aligned_bounding_box()
    center=bbox.get_center(); radius=2.5*max(bbox.get_extent())

    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("obj",geom,make_material(geom))

    azs = np.arange(baz-FINE_AZ_DELTA, baz+FINE_AZ_DELTA+1e-6, FINE_AZ_STEP)
    els = np.arange(bel-FINE_EL_DELTA, bel+FINE_EL_DELTA+1e-6, FINE_EL_STEP)

    fine_best = (-1,None,None,None)
    for el in els:
        for az in azs:
            s = silhouette_score(sketch, normalize_mask(render_silhouette(renderer,center,radius,az,el), IMG_SIZE))
            if s>fine_best[0]:
                fine_best=(s,az,el,True)
                print(f"  New fine best: score={s:.4f} | az={az:.1f}, el={el:.1f}")

    best_score, az, el,_ = fine_best
    print("\n🎯 FINAL BEST:"
          f"\n  az = {az:.1f}, el = {el:.1f}, score = {best_score:.4f}")

    rgb,sil = render_rgb_and_sil(renderer,center,radius,az,el)
    cv2.imwrite(str(SHAPE_IMG), cv2.resize(rgb,(IMG_SIZE,IMG_SIZE)))
    save_overlay(sketch, normalize_mask(sil,IMG_SIZE), OVERLAY_IMG)

    json.dump({
        "base_rotation":R.tolist(),
        "azimuth_deg":float(az),
        "elevation_deg":float(el),
        "radius":float(radius),
        "center":[float(c) for c in center],
        "score":float(best_score),
    }, CAMERA_JSON.open("w"),indent=2)

    print("\n📁 Results saved to:", OUT_DIR)
    print("   - shape.png\n   - overlay.png\n   - camera.json")
    print("\n✨ Done!")


if __name__=="__main__":
    main()
