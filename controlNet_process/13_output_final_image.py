#!/usr/bin/env python3
"""
paste_back_apply_translations.py

Apply 2D paste-back results driven by per-label XY translations (pixels),
using:
  - ORIGINAL masks/components from:
      sketch/segmentation_original_image/view_{x}/{label}_mask.png
      sketch/segmentation_original_image/view_{x}/{label}.png
  - WARPED (translated) masks from:
      sketch/back_project_masks/view_{x}/paste_back/paste_back_results.json
      (path(s) inside JSON: outputs.masks_dir)

Outputs per view:
  sketch/final_outputs/view_{x}/
    old.png
    removed.png
    new.png
    verification/
      {label}_overlay.png
      all_overlays.png
      all_after_components.png

Notes:
- We remove "old" regions using the ORIGINAL mask (erode 5px then dilate remove_radius_px).
- We paste "after" components by translating the ORIGINAL component image by translations_xy[label],
  and gating its alpha by the WARPED mask in outputs.masks_dir/{label}_mask.png.
"""

import os
import json
import glob
from typing import Dict, Any, Tuple

import cv2
import numpy as np


# -------------------------
# IO helpers
# -------------------------

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


# -------------------------
# Morphology
# -------------------------

def _dilate_mask(mask01: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask01
    k = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask01, kernel, iterations=1)


def _erode_mask(mask01: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask01
    k = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask01, kernel, iterations=1)


# -------------------------
# Alpha / compositing
# -------------------------

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


def _alpha_over_rgba(bg_rgba: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    """Porter-Duff "over" for RGBA uint8."""
    bg = bg_rgba.astype(np.float32) / 255.0
    fg = fg_rgba.astype(np.float32) / 255.0

    fg_a = fg[:, :, 3:4]
    bg_a = bg[:, :, 3:4]

    out_a = fg_a + bg_a * (1.0 - fg_a)
    out_rgb = (fg[:, :, :3] * fg_a + bg[:, :, :3] * bg_a * (1.0 - fg_a)) / np.clip(out_a, 1e-6, 1.0)

    out = np.concatenate([out_rgb, out_a], axis=2)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _composite_rgba_over_bgr(base_bgr: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    """Composite fg_rgba over base_bgr (fg alpha already encodes where to draw)."""
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
# Translation warp
# -------------------------

def _warp_translate_rgba(src_rgba: np.ndarray, dx: float, dy: float, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    Translate in image pixel coordinates:
      x' = x + dx
      y' = y + dy

    Uses borderValue=(0,0,0,0) so transparent outside.
    """
    Ht, Wt = out_hw
    M = np.array([[1.0, 0.0, float(dx)],
                  [0.0, 1.0, float(dy)]], dtype=np.float32)
    return cv2.warpAffine(
        src_rgba,
        M,
        (Wt, Ht),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


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
    before_blue = _colorize_ink_layer(
        ink=before_ink,
        region01=remove_mask01.astype(np.uint8),
        color="blue",
        alpha_fill=0.40,
        alpha_edge=0.95,
        edge_thickness=1,
    )

    # AFTER (red): pasted-back delta (after_comp_rgba alpha)
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
# Path helpers for paste_back_results.json
# -------------------------

def _resolve_outputs_dir(pb_json_path: str, outputs_dir_from_json: str) -> str:
    """
    In paste_back_results.json, outputs.masks_dir is often relative, like "masks/".
    Interpret relative paths as relative to the folder containing paste_back_results.json.
    """
    if not outputs_dir_from_json:
        raise ValueError("paste_back_results.json missing outputs.masks_dir")
    if os.path.isabs(outputs_dir_from_json):
        return outputs_dir_from_json
    base = os.path.dirname(pb_json_path)
    return os.path.normpath(os.path.join(base, outputs_dir_from_json))


def _find_warped_mask_path(masks_dir: str, label: str) -> str:
    """
    Your warped masks are expected as:
      {masks_dir}/{label}_mask.png

    If you used a different naming convention, edit this function.
    """
    cand = os.path.join(masks_dir, f"{label}_mask.png")
    if os.path.exists(cand):
        return cand

    # fallback: try "{label}.png" (just in case)
    cand2 = os.path.join(masks_dir, f"{label}.png")
    if os.path.exists(cand2):
        return cand2

    raise FileNotFoundError(f"Cannot find warped mask for {label} in {masks_dir}")


# -------------------------
# Main pipeline per view
# -------------------------

def apply_view_from_paste_back(pb_json_path: str, out_root: str, remove_radius_px: int = 5) -> None:
    data = _read_json(pb_json_path)

    view_name = data.get("view", os.path.basename(os.path.dirname(os.path.dirname(pb_json_path))))
    view_id = view_name.replace("view_", "").strip()

    # view image
    view_img_candidates = [
        os.path.join("sketch", "view", f"view_{view_id}.png"),
        os.path.join("sketch", "views", f"view_{view_id}.png"),
    ]
    view_img_path = next((p for p in view_img_candidates if p and os.path.exists(p)), None)
    if view_img_path is None:
        raise FileNotFoundError(f"Cannot find view image for {view_name}. Tried: {view_img_candidates}")

    base_bgr = _imread_rgb(view_img_path)
    Ht, Wt = base_bgr.shape[:2]

    # segmentation dir (original masks/components)
    seg_dir_candidates = [
        os.path.join("sketch", "segmentation_original_image", view_name),
        os.path.join("sketch", "segmentation_original_image", f"view_{view_id}"),
    ]
    seg_dir = next((d for d in seg_dir_candidates if os.path.isdir(d)), None)
    if seg_dir is None:
        raise FileNotFoundError(f"Cannot find segmentation dir for {view_name}. Tried: {seg_dir_candidates}")

    # warped masks dir from paste_back_results.json
    outputs = data.get("outputs", {})
    masks_dir = _resolve_outputs_dir(pb_json_path, outputs.get("masks_dir", ""))

    # translations
    translations_xy = data.get("translations_xy", {})
    labels_involved = data.get("labels_involved", list(translations_xy.keys()))

    removed_bgr = base_bgr.copy()
    all_after_rgba = np.zeros((Ht, Wt, 4), dtype=np.uint8)

    out_dir = os.path.join(out_root, view_name)
    os.makedirs(out_dir, exist_ok=True)
    verif_dir = os.path.join(out_dir, "verification")
    os.makedirs(verif_dir, exist_ok=True)

    all_overlay_rgba = np.zeros((Ht, Wt, 4), dtype=np.uint8)

    for label in labels_involved:
        # require original assets
        mask_path = os.path.join(seg_dir, f"{label}_mask.png")
        comp_img_path = os.path.join(seg_dir, f"{label}.png")

        if not os.path.exists(mask_path):
            # some labels might not have masks (depending on pipeline); skip quietly
            continue
        if not os.path.exists(comp_img_path):
            continue

        # warped mask path
        warped_mask_path = _find_warped_mask_path(masks_dir, label)

        # dx, dy translation (pixels)
        dx_dy = translations_xy.get(label, [0.0, 0.0])
        dx, dy = float(dx_dy[0]), float(dx_dy[1])

        # --- removal mask (in view space) from ORIGINAL mask ---
        old_mask01 = _imread_mask01(mask_path)
        old_mask01 = cv2.resize(old_mask01, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

        # shrink inward by 5px so we remove LESS of the old area
        old_mask01_shrunk = _erode_mask(old_mask01, 5)

        # then outward tolerance (default 5px)
        remove_mask01 = _dilate_mask(old_mask01_shrunk, remove_radius_px)

        removed_bgr[remove_mask01.astype(bool)] = (255, 255, 255)

        # --- load component + ensure alpha from ORIGINAL mask ---
        comp_rgba = _imread_rgba(comp_img_path)
        comp_mask_local01 = _imread_mask01(mask_path)
        comp_rgba = _ensure_component_rgba(comp_rgba, comp_mask_local01)

        # --- translate component by (dx, dy) into view space ---
        moved_rgba = _warp_translate_rgba(comp_rgba, dx=dx, dy=dy, out_hw=(Ht, Wt))

        # --- warped mask (already in view space) ---
        new_mask01 = _imread_mask01(warped_mask_path)
        new_mask01 = cv2.resize(new_mask01, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

        # --- build per-label AFTER delta (RGBA): alpha = moved_alpha * new_mask01 ---
        after_comp = moved_rgba.copy()
        after_comp[:, :, 3] = (after_comp[:, :, 3].astype(np.uint8) * new_mask01.astype(np.uint8))

        # accumulate all after components into ONE transparent layer
        all_after_rgba = _alpha_over_rgba(all_after_rgba, after_comp)

        # per-label verification overlay (single png)
        label_overlay_rgba = _make_label_overlay_rgba(
            base_bgr=base_bgr,
            remove_mask01=remove_mask01,
            after_comp_rgba=after_comp,
        )
        cv2.imwrite(os.path.join(verif_dir, f"{label}_overlay.png"), _rgba_on_white(label_overlay_rgba))
        all_overlay_rgba = _alpha_over_rgba(all_overlay_rgba, label_overlay_rgba)

    # new.png = removed.png + all_after_rgba
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

    pb_paths = sorted(glob.glob(os.path.join("sketch", "back_project_masks", "view_*", "paste_back", "paste_back_results.json")))
    if not pb_paths:
        raise FileNotFoundError("No paste_back_results.json found under sketch/back_project_masks/view_*/paste_back/paste_back_results.json")

    for p in pb_paths:
        apply_view_from_paste_back(p, out_root=out_root, remove_radius_px=5)


if __name__ == "__main__":
    main()
