#!/usr/bin/env python3
# 14_image_fix_local_fluxfill.py
#
# Local FLUX.1-Fill-dev inpainting:
# - for each sketch/final_outputs/view_x/new.png
# - generate seam_mask.png (from sketch/back_project_masks/view_x/mask_warps.json -> outputs.mask_new)
# - run FluxFillPipeline inpaint
# - save corrected.png in the same folder
#
# NO BFL_API_KEY. No requests.

import os
import json
import glob
from typing import List

import cv2
import numpy as np

import torch
from PIL import Image
from diffusers import FluxFillPipeline


PROMPT = (
    "Repair compositing artifacts caused by translating/scaling parts. "
    "Only edit pixels inside the mask. Outside the mask, keep the image identical. "
    "Inside the mask, fix boundary seams/gaps/overlaps/jagged edges and match neighboring shading/edges. "
    "Do NOT move any components, do NOT change layout, do NOT add/remove parts."
)

# Tuning knobs (safe defaults for seam repair)
R_BAND = 5          # seam band thickness in pixels
BLUR_SIGMA = 1.0    # feather mask edge slightly
STEPS = 30
GUIDANCE = 30
MAX_SEQ_LEN = 512


def read_gray(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m


def write_png(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def load_masknew_paths_for_view(view_name: str) -> List[str]:
    warp_json = os.path.join("sketch", "back_project_masks", view_name, "mask_warps.json")
    if not os.path.exists(warp_json):
        return []

    with open(warp_json, "r") as f:
        data = json.load(f)

    out = []
    for _, entry in (data.get("labels", {}) or {}).items():
        if not entry.get("has_mask", False):
            continue
        p = ((entry.get("outputs", {}) or {}).get("mask_new", "") or "").strip()
        if p and os.path.exists(p):
            out.append(p)
    return out


def make_seam_mask_from_masknew(mask_new_paths: List[str], H: int, W: int, r: int, blur_sigma: float) -> np.ndarray:
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    seam = np.zeros((H, W), dtype=np.uint8)

    for p in mask_new_paths:
        m = read_gray(p)
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m01 = (m > 127).astype(np.uint8)

        dil = cv2.dilate(m01, kernel, iterations=1)
        ero = cv2.erode(m01, kernel, iterations=1)
        band = ((dil - ero) > 0).astype(np.uint8)

        seam = np.maximum(seam, band)

    seam = (seam * 255).astype(np.uint8)

    if blur_sigma and blur_sigma > 0:
        seam_f = seam.astype(np.float32) / 255.0
        seam_f = cv2.GaussianBlur(seam_f, (0, 0), blur_sigma)
        seam = np.clip(seam_f * 255.0, 0, 255).astype(np.uint8)

    return seam


def pil_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_mask(path: str) -> Image.Image:
    # white=inpaint, black=keep
    return Image.open(path).convert("L")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA not found. FLUX fill is heavy; please run on a GPU node.")

    # Model card recommends bfloat16.
    pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    torch_dtype=torch.bfloat16,
    )

    # Key: don't do .to("cuda"); offload instead
    pipe.enable_model_cpu_offload()        # moves modules to GPU only when needed
    pipe.enable_attention_slicing()        # reduces peak mem


    view_dirs = sorted(glob.glob(os.path.join("sketch", "final_outputs", "view_*")))
    if not view_dirs:
        raise FileNotFoundError("No folders found: sketch/final_outputs/view_*")

    for folder in view_dirs:
        view_name = os.path.basename(folder)
        new_png = os.path.join(folder, "new.png")
        if not os.path.exists(new_png):
            continue

        img_bgr = cv2.imread(new_png, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(new_png)
        H, W = img_bgr.shape[:2]

        mask_new_paths = load_masknew_paths_for_view(view_name)
        if not mask_new_paths:
            print(f"[skip] {view_name}: no mask_new paths found in mask_warps.json")
            continue

        seam_mask = make_seam_mask_from_masknew(mask_new_paths, H, W, r=R_BAND, blur_sigma=BLUR_SIGMA)
        seam_mask_path = os.path.join(folder, "seam_mask.png")
        write_png(seam_mask_path, seam_mask)

        image = pil_rgb(new_png)
        mask = pil_mask(seam_mask_path)

        out = pipe(
            prompt=PROMPT,
            image=image,
            mask_image=mask,
            height=H,
            width=W,
            guidance_scale=GUIDANCE,
            num_inference_steps=STEPS,
            max_sequence_length=MAX_SEQ_LEN,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        out_path = os.path.join(folder, "corrected.png")
        out.save(out_path)
        print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
