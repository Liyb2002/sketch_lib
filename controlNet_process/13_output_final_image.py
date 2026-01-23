#!/usr/bin/env python3
import os
import json
import glob
from typing import Dict, Any, Tuple

import cv2
import numpy as np


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _imread_rgba(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 4:
        pass
    else:
        raise ValueError(f"Unexpected channels in {path}: {img.shape}")
    return img


def _imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _imread_mask01(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8)


def _dilate_mask(mask01: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask01
    k = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask01, kernel, iterations=1)


def _ensure_component_rgba(component_png: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    comp = component_png.copy()
    if comp.shape[2] == 4:
        alpha = comp[:, :, 3]
        if np.max(alpha) == 0:
            comp[:, :, 3] = (mask01 * 255).astype(np.uint8)
        return comp
    comp = cv2.cvtColor(comp, cv2.COLOR_BGR2BGRA)
    comp[:, :, 3] = (mask01 * 255).astype(np.uint8)
    return comp


def _warp_rgba(src_rgba: np.ndarray, H: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    Ht, Wt = out_hw
    return cv2.warpPerspective(
        src_rgba,
        H,
        (Wt, Ht),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _composite_rgb_over(base_bgr: np.ndarray, fg_rgba: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    base = base_bgr.astype(np.float32)
    fg_rgb = fg_rgba[:, :, :3].astype(np.float32)

    alpha_from_rgba = fg_rgba[:, :, 3].astype(np.float32) / 255.0
    alpha = np.clip(alpha_from_rgba * mask01.astype(np.float32), 0.0, 1.0)[:, :, None]

    out = fg_rgb * alpha + base * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_view(mask_warps_json: str, out_root: str, remove_radius_px: int = 5) -> None:
    data = _read_json(mask_warps_json)

    view_name = data.get("view", os.path.basename(os.path.dirname(mask_warps_json)))
    view_id = view_name.replace("view_", "").strip()

    view_img_candidates = [
        os.path.join("sketch", "view", f"view_{view_id}.png"),
        os.path.join("sketch", "views", f"view_{view_id}.png"),
        data.get("image", ""),
    ]
    view_img_path = next((p for p in view_img_candidates if p and os.path.exists(p)), None)
    if view_img_path is None:
        raise FileNotFoundError(f"Cannot find view image for {view_name}. Tried: {view_img_candidates}")

    base_bgr = _imread_rgb(view_img_path)
    Ht, Wt = base_bgr.shape[:2]

    removed_bgr = base_bgr.copy()
    new_bgr = base_bgr.copy()

    labels: Dict[str, Any] = data.get("labels", {})
    for comp_name, entry in labels.items():
        if not entry.get("has_mask", False):
            continue

        seg_dir_candidates = [
            os.path.join("sketch", "segmentation_original_image", view_name),
            os.path.join("sketch", "segmentation_original_image", f"view_{view_id}"),
        ]
        seg_dir = next((d for d in seg_dir_candidates if os.path.isdir(d)), None)
        if seg_dir is None:
            raise FileNotFoundError(f"Cannot find segmentation dir for {view_name}")

        mask_path = entry.get("mask_path", os.path.join(seg_dir, f"{comp_name}_mask.png"))
        if not os.path.exists(mask_path):
            raise FileNotFoundError(mask_path)

        comp_img_path = os.path.join(seg_dir, f"{comp_name}.png")
        if not os.path.exists(comp_img_path):
            raise FileNotFoundError(comp_img_path)

        outputs = entry.get("outputs", {})
        new_mask_path = outputs.get("mask_new", "")
        if not new_mask_path or not os.path.exists(new_mask_path):
            raise FileNotFoundError(new_mask_path if new_mask_path else f"(missing outputs.mask_new) for {comp_name}")

        Hmat = np.array(entry.get("homography_before_to_after", None), dtype=np.float64)
        if Hmat.shape != (3, 3):
            raise ValueError(f"Bad homography for {comp_name}")

        old_mask01 = _imread_mask01(mask_path)
        old_mask01 = cv2.resize(old_mask01, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        remove_mask01 = _dilate_mask(old_mask01, remove_radius_px)

        removed_bgr[remove_mask01.astype(bool)] = (255, 255, 255)
        new_bgr[remove_mask01.astype(bool)] = (255, 255, 255)

        comp_rgba = _imread_rgba(comp_img_path)
        comp_mask_local01 = _imread_mask01(mask_path)
        comp_rgba = _ensure_component_rgba(comp_rgba, comp_mask_local01)

        warped_rgba = _warp_rgba(comp_rgba, Hmat, (Ht, Wt))

        new_mask01 = _imread_mask01(new_mask_path)
        new_mask01 = cv2.resize(new_mask01, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

        new_bgr = _composite_rgb_over(new_bgr, warped_rgba, new_mask01)

    out_dir = os.path.join(out_root, view_name)
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "old.png"), base_bgr)
    cv2.imwrite(os.path.join(out_dir, "removed.png"), removed_bgr)
    cv2.imwrite(os.path.join(out_dir, "new.png"), new_bgr)


def main():
    out_root = os.path.join("sketch", "final_outputs")

    warp_paths = sorted(glob.glob(os.path.join("sketch", "back_project_masks", "view_*", "mask_warps.json")))
    if not warp_paths:
        raise FileNotFoundError("No mask_warps.json found under sketch/back_project_masks/view_*/mask_warps.json")

    for p in warp_paths:
        apply_view(p, out_root=out_root, remove_radius_px=5)


if __name__ == "__main__":
    main()
