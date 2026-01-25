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
    """
    Composite fg_rgba over base_bgr, but ONLY where mask01==1.
    Uses fg_rgba alpha * mask01.
    """
    base = base_bgr.astype(np.float32)
    fg_rgb = fg_rgba[:, :, :3].astype(np.float32)

    alpha_from_rgba = fg_rgba[:, :, 3].astype(np.float32) / 255.0
    alpha = np.clip(alpha_from_rgba * mask01.astype(np.float32), 0.0, 1.0)[:, :, None]

    out = fg_rgb * alpha + base * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------
# RGBA delta layer compositing (fixes new.png correctness)
# -------------------------

def _alpha_over_rgba(bg_rgba: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    """
    Porter-Duff "over" for RGBA uint8.
    """
    bg = bg_rgba.astype(np.float32) / 255.0
    fg = fg_rgba.astype(np.float32) / 255.0

    fg_a = fg[:, :, 3:4]
    bg_a = bg[:, :, 3:4]

    out_a = fg_a + bg_a * (1.0 - fg_a)
    out_rgb = (fg[:, :, :3] * fg_a + bg[:, :, :3] * bg_a * (1.0 - fg_a)) / np.clip(out_a, 1e-6, 1.0)

    out = np.concatenate([out_rgb, out_a], axis=2)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _composite_rgba_over_bgr(base_bgr: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    """
    Composite fg_rgba over base_bgr (no extra mask; fg_rgba alpha already encodes where to draw).
    """
    base = base_bgr.astype(np.float32)
    fg = fg_rgba[:, :, :3].astype(np.float32)
    a = (fg_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

    out = fg * a + base * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _rgba_on_white(rgba: np.ndarray) -> np.ndarray:
    H, W = rgba.shape[:2]
    bg = np.full((H, W, 3), 255, dtype=np.uint8)

    a = (rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    fg = rgba[:, :, :3].astype(np.float32)
    out = fg * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------
# Verification rendering (blue before, red after)
# -------------------------

def _ink_from_bgr(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = (255 - gray).astype(np.uint8)
    return ink


def _ink_from_rgba(rgba: np.ndarray) -> np.ndarray:
    return _ink_from_bgr(rgba[:, :, :3])


def _mask_edge(mask01: np.ndarray, thickness: int = 1) -> np.ndarray:
    if thickness <= 0:
        thickness = 1
    k = 2 * thickness + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(mask01, kernel, iterations=1)
    ero = cv2.erode(mask01, kernel, iterations=1)
    edge = (dil ^ ero).astype(np.uint8)
    return edge


def _colorize_ink_layer(
    ink: np.ndarray,
    region01: np.ndarray,
    color: str,
    alpha_fill: float = 0.45,
    alpha_edge: float = 0.95,
    edge_thickness: int = 1,
) -> np.ndarray:
    H, W = ink.shape[:2]
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    ink_f = ink.astype(np.float32) / 255.0
    region_f = region01.astype(np.float32)

    a_fill = (alpha_fill * ink_f * region_f * 255.0).astype(np.uint8)

    if color == "blue":
        rgba[:, :, 0] = (ink_f * 255.0).astype(np.uint8)  # B
        rgba[:, :, 1] = 0
        rgba[:, :, 2] = 0
    elif color == "red":
        rgba[:, :, 0] = 0
        rgba[:, :, 1] = 0
        rgba[:, :, 2] = (ink_f * 255.0).astype(np.uint8)  # R
    else:
        raise ValueError("color must be 'blue' or 'red'")

    rgba[:, :, 3] = a_fill

    edge01 = _mask_edge(region01, thickness=edge_thickness)
    if np.any(edge01):
        a_edge = int(alpha_edge * 255.0)
        if color == "blue":
            rgba[edge01.astype(bool), 0] = 255
            rgba[edge01.astype(bool), 1] = 0
            rgba[edge01.astype(bool), 2] = 0
        else:
            rgba[edge01.astype(bool), 0] = 0
            rgba[edge01.astype(bool), 1] = 0
            rgba[edge01.astype(bool), 2] = 255
        rgba[edge01.astype(bool), 3] = np.maximum(rgba[edge01.astype(bool), 3], a_edge).astype(np.uint8)

    return rgba


def _make_label_overlay_rgba(
    base_bgr: np.ndarray,
    remove_mask01: np.ndarray,
    after_comp_rgba: np.ndarray,
) -> np.ndarray:
    # BEFORE (blue): original pixels inside remove mask
    before_ink = _ink_from_bgr(base_bgr)
    before_region01 = remove_mask01.astype(np.uint8)
    before_blue = _colorize_ink_layer(
        ink=before_ink,
        region01=before_region01,
        color="blue",
        alpha_fill=0.40,
        alpha_edge=0.95,
        edge_thickness=1,
    )

    # AFTER (red): the actual pasted-back delta layer (after_comp_rgba alpha)
    post_region01 = (after_comp_rgba[:, :, 3] > 0).astype(np.uint8)
    post_ink = _ink_from_rgba(after_comp_rgba)
    post_red = _colorize_ink_layer(
        ink=post_ink,
        region01=post_region01,
        color="red",
        alpha_fill=0.55,
        alpha_edge=0.98,
        edge_thickness=1,
    )

    overlay = np.zeros_like(before_blue)
    overlay = _alpha_over_rgba(overlay, before_blue)
    overlay = _alpha_over_rgba(overlay, post_red)  # red over blue
    return overlay


# -------------------------
# Main pipeline
# -------------------------

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

    # We'll compute:
    #  - removed_bgr: base with ALL old regions removed (union over labels)
    #  - all_after_rgba: transparent delta containing ALL warped components at new locations
    removed_bgr = base_bgr.copy()
    all_after_rgba = np.zeros((Ht, Wt, 4), dtype=np.uint8)

    out_dir = os.path.join(out_root, view_name)
    os.makedirs(out_dir, exist_ok=True)

    verif_dir = os.path.join(out_dir, "verification")
    os.makedirs(verif_dir, exist_ok=True)

    all_overlay_rgba = np.zeros((Ht, Wt, 4), dtype=np.uint8)

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

        # --- removal mask (in view space) ---
        old_mask01 = _imread_mask01(mask_path)
        old_mask01 = cv2.resize(old_mask01, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        remove_mask01 = _dilate_mask(old_mask01, remove_radius_px)
        removed_bgr[remove_mask01.astype(bool)] = (255, 255, 255)

        # --- load + warp component rgba ---
        comp_rgba = _imread_rgba(comp_img_path)
        comp_mask_local01 = _imread_mask01(mask_path)
        comp_rgba = _ensure_component_rgba(comp_rgba, comp_mask_local01)

        warped_rgba = _warp_rgba(comp_rgba, Hmat, (Ht, Wt))

        # --- new mask (in view space) ---
        new_mask01 = _imread_mask01(new_mask_path)
        new_mask01 = cv2.resize(new_mask01, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

        # --- build per-component AFTER delta layer (RGBA) ---
        # alpha = warped_alpha * new_mask01 (new_mask01 is 0/1)
        after_comp = warped_rgba.copy()
        after_comp[:, :, 3] = (after_comp[:, :, 3].astype(np.uint8) * new_mask01.astype(np.uint8))

        # accumulate all after components into ONE transparent layer
        all_after_rgba = _alpha_over_rgba(all_after_rgba, after_comp)

        # (1) per-label verification: save ONLY one overlay png
        label_overlay_rgba = _make_label_overlay_rgba(
            base_bgr=base_bgr,
            remove_mask01=remove_mask01,
            after_comp_rgba=after_comp,
        )
        cv2.imwrite(os.path.join(verif_dir, f"{comp_name}_overlay.png"), _rgba_on_white(label_overlay_rgba))

        # accumulate "all overlays" (visual only)
        all_overlay_rgba = _alpha_over_rgba(all_overlay_rgba, label_overlay_rgba)

    # Now new.png is EXACTLY removed.png + all_after_rgba
    new_bgr = _composite_rgba_over_bgr(removed_bgr, all_after_rgba)

    # Save core outputs
    cv2.imwrite(os.path.join(out_dir, "old.png"), base_bgr)
    cv2.imwrite(os.path.join(out_dir, "removed.png"), removed_bgr)
    cv2.imwrite(os.path.join(out_dir, "new.png"), new_bgr)

    # Save verification aggregates
    cv2.imwrite(os.path.join(verif_dir, "all_overlays.png"), _rgba_on_white(all_overlay_rgba))
    cv2.imwrite(os.path.join(verif_dir, "all_after_components.png"), _rgba_on_white(all_after_rgba))


def main():
    out_root = os.path.join("sketch", "final_outputs")

    warp_paths = sorted(glob.glob(os.path.join("sketch", "back_project_masks", "view_*", "mask_warps.json")))
    if not warp_paths:
        raise FileNotFoundError("No mask_warps.json found under sketch/back_project_masks/view_*/mask_warps.json")

    for p in warp_paths:
        apply_view(p, out_root=out_root, remove_radius_px=5)


if __name__ == "__main__":
    main()
