#!/usr/bin/env python3
# 14_sdxl_inpaint_tune_and_debug.py
#
# Saves:
#   sketch/final_outputs/view_x/inpainting_test/
#     input.png
#     mask.png
#     output.png
#     masked_before.png
#     masked_after.png
#     masked_diff.png
#
# And also writes:
#   sketch/final_outputs/view_x/fix/sdxl_inpaint.png
#   sketch/final_outputs/view_x/sdxl_inpaint.png

import os
import glob
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
from diffusers import AutoPipelineForInpainting


MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

PROMPT = (
    "Seam and gap repair (technical). In the masked region, CONNECT the pasted parts: "
    "extend edges/contours across gaps, fill missing pixels, close small holes, and make boundaries continuous. "
    "Synthesize only what is needed to connect smoothly (no redesign). "
    "Match local color/texture/shading so the join looks naturally continuous. "
    "Outside the mask must remain unchanged."
)

NEG_PROMPT = (
    "text, watermark, logo, letters, numbers, symbols, "
    "extra objects, extra parts, new components, "
    "global color shift, global lighting change, "
    "moving objects outside mask, changing layout outside mask"
)

# ---- TUNE THESE ----
NUM_STEPS = 25     # try 35-42
GUIDANCE = 7     # try 5.0-6.5
STRENGTH = 0.9    # try 0.68-0.78
SEED = 42
# --------------------


def _read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _read_mask_u8(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = (m > 10).astype(np.uint8) * 255
    return m.astype(np.uint8)


def _bgr_to_pil_rgb(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB")


def _mask_to_pil_l(mask_u8: np.ndarray) -> Image.Image:
    return Image.fromarray(mask_u8, mode="L")


def _nearest_multiple(x: int, base: int = 8) -> int:
    return max(base, int(round(x / base)) * base)


def _prep_size(img_bgr: np.ndarray, mask_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    H, W = img_bgr.shape[:2]
    if mask_u8.shape[:2] != (H, W):
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    H2 = _nearest_multiple(H, 8)
    W2 = _nearest_multiple(W, 8)
    if (H2, W2) != (H, W):
        img2 = cv2.resize(img_bgr, (W2, H2), interpolation=cv2.INTER_LINEAR)
        mask2 = cv2.resize(mask_u8, (W2, H2), interpolation=cv2.INTER_NEAREST)
        return img2, mask2, (H, W)

    return img_bgr, mask_u8, (H, W)


def _save_bgr(path: str, img_bgr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, img_bgr):
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def _save_gray(path: str, img_u8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, img_u8):
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def _pick_input_image(view_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(view_dir, "fix", "new.png"),
        os.path.join(view_dir, "new.png"),
        os.path.join(view_dir, "gemini_fix", "new.png"),
        os.path.join(view_dir, "fix", "input.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    device = "cuda"
    dtype = torch.float16

    pipe = AutoPipelineForInpainting.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16",
    ).to(device)

    # memory helpers
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    gen = torch.Generator(device=device).manual_seed(SEED)

    view_dirs = sorted(glob.glob(os.path.join("sketch", "final_outputs", "view_*")))
    if not view_dirs:
        raise FileNotFoundError("No view folders found: sketch/final_outputs/view_*")

    for view_dir in view_dirs:
        view_name = os.path.basename(view_dir)

        mask_path = os.path.join(view_dir, "fix", "diff_sum_mask.png")
        if not os.path.exists(mask_path):
            continue

        img_path = _pick_input_image(view_dir)
        if img_path is None:
            print(f"[skip] {view_name}: cannot find input image")
            continue

        out_dir = os.path.join(view_dir, "inpainting_test")
        os.makedirs(out_dir, exist_ok=True)

        img_bgr = _read_bgr(img_path)
        mask_u8 = _read_mask_u8(mask_path)

        _save_bgr(os.path.join(out_dir, "input.png"), img_bgr)
        _save_gray(os.path.join(out_dir, "mask.png"), mask_u8)

        img_bgr2, mask_u82, orig_hw = _prep_size(img_bgr, mask_u8)

        image_pil = _bgr_to_pil_rgb(img_bgr2)
        mask_pil = _mask_to_pil_l(mask_u82)

        out = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE,
            strength=STRENGTH,
            generator=gen,
        ).images[0]

        out_rgb = np.array(out.convert("RGB"))
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        H0, W0 = orig_hw
        if out_bgr.shape[:2] != (H0, W0):
            out_bgr = cv2.resize(out_bgr, (W0, H0), interpolation=cv2.INTER_LINEAR)

        _save_bgr(os.path.join(out_dir, "output.png"), out_bgr)
        _save_bgr(os.path.join(view_dir, "fix", "sdxl_inpaint.png"), out_bgr)
        _save_bgr(os.path.join(view_dir, "sdxl_inpaint.png"), out_bgr)

        # ---- save before/after inside mask ----
        m = (mask_u8 > 127).astype(np.uint8)
        m3 = np.repeat(m[:, :, None], 3, axis=2)

        masked_before = (img_bgr * m3).astype(np.uint8)
        masked_after = (out_bgr * m3).astype(np.uint8)
        diff = cv2.absdiff(img_bgr, out_bgr)
        masked_diff = (diff * m3).astype(np.uint8)

        _save_bgr(os.path.join(out_dir, "masked_before.png"), masked_before)
        _save_bgr(os.path.join(out_dir, "masked_after.png"), masked_after)
        _save_bgr(os.path.join(out_dir, "masked_diff.png"), masked_diff)

        print(f"[OK] {view_name} | strength={STRENGTH} guidance={GUIDANCE} steps={NUM_STEPS}")

    print("Done.")


if __name__ == "__main__":
    main()
