#!/usr/bin/env python3
# debug_save_pre_post_and_diff_masks.py
#
# Saves per-label pre/post/diff masks and aggregates.
# Computes:
#   - diff_sum_mask.png (OR over thickened diffs)
#   - sum_after_mask.png (OR over post masks)
#   - sum_after_count.npy (per-pixel count of how many post masks cover it)
#   - sum_after_count.png (a 0..255 visualization of the count map)
#   - sum_after_plus_diff_sum_mask.png (OR(sum_after_mask, diff_sum_mask))

import os
import json
import glob
from typing import Dict, Any

import cv2
import numpy as np


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


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


def _thicken_both_ways(mask01: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Symmetric thickening around boundary:
      thick = mask ∪ (dilate(mask) XOR erode(mask))
    """
    if radius_px <= 0:
        return mask01

    k = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    dil = cv2.dilate(mask01, kernel, iterations=1)
    ero = cv2.erode(mask01, kernel, iterations=1)

    band = (dil ^ ero).astype(np.uint8)
    thick = np.maximum(mask01, band)
    return thick


def _mask01_to_png(mask01: np.ndarray) -> np.ndarray:
    return (mask01.astype(np.uint8) * 255)


def _infer_view_image_path(view_name: str, view_id: str, data: Dict[str, Any]) -> str:
    candidates = [
        os.path.join("sketch", "view", f"view_{view_id}.png"),
        os.path.join("sketch", "views", f"view_{view_id}.png"),
        data.get("image", ""),
    ]
    p = next((c for c in candidates if c and os.path.exists(c)), None)
    if p is None:
        raise FileNotFoundError(f"Cannot find view image for {view_name}. Tried: {candidates}")
    return p


def _infer_seg_dir(view_name: str, view_id: str) -> str:
    candidates = [
        os.path.join("sketch", "segmentation_original_image", view_name),
        os.path.join("sketch", "segmentation_original_image", f"view_{view_id}"),
    ]
    d = next((c for c in candidates if os.path.isdir(c)), None)
    if d is None:
        raise FileNotFoundError(f"Cannot find segmentation dir for {view_name}. Tried: {candidates}")
    return d


def _summarize_label_edit(entry: Dict[str, Any]) -> str:
    parts = []
    if "edit_type" in entry:
        parts.append(f"edit_type={entry.get('edit_type')}")
    if "outputs" in entry and isinstance(entry["outputs"], dict):
        parts.append(f"mask_new={'yes' if entry['outputs'].get('mask_new') else 'no'}")
    if "homography_before_to_after" in entry:
        parts.append("has_H")
    return ", ".join(parts) if parts else "(no explicit edit fields)"


def _count_to_vis_png(count_map: np.ndarray) -> np.ndarray:
    """
    Visualize count map as 8-bit PNG.
    Scales [0..max] to [0..255]. If max==0, returns zeros.
    """
    mx = int(count_map.max()) if count_map.size else 0
    if mx <= 0:
        return np.zeros_like(count_map, dtype=np.uint8)
    vis = (count_map.astype(np.float32) / float(mx) * 255.0).round().astype(np.uint8)
    return vis


def process_view(mask_warps_json: str, out_root: str, remove_radius_px: int = 5, diff_thicken_px: int = 5) -> None:
    data = _read_json(mask_warps_json)

    view_name = data.get("view", os.path.basename(os.path.dirname(mask_warps_json)))
    view_id = view_name.replace("view_", "").strip()

    view_img_path = _infer_view_image_path(view_name, view_id, data)
    base_bgr = _imread_rgb(view_img_path)
    Ht, Wt = base_bgr.shape[:2]

    labels: Dict[str, Any] = data.get("labels", {})
    edited = [(k, v) for k, v in labels.items() if isinstance(v, dict) and v.get("has_mask", False)]

    print("\n" + "=" * 80)
    print(f"[{view_name}]")
    print(f"mask_warps: {mask_warps_json}")
    print(f"view image: {view_img_path}  (H,W)=({Ht},{Wt})")
    print(f"edited labels: {len(edited)}")
    for name, entry in edited:
        print(f"  - {name}: {_summarize_label_edit(entry)}")

    out_dir = os.path.join(out_root, view_name, "fix")
    os.makedirs(out_dir, exist_ok=True)

    seg_dir = _infer_seg_dir(view_name, view_id)

    diff_sum01 = np.zeros((Ht, Wt), dtype=np.uint8)

    # NEW aggregates for "after"
    sum_after01 = np.zeros((Ht, Wt), dtype=np.uint8)      # binary union of post masks
    sum_after_count = np.zeros((Ht, Wt), dtype=np.uint16) # per-pixel counts

    for comp_name, entry in edited:
        mask_path = entry.get("mask_path", os.path.join(seg_dir, f"{comp_name}_mask.png"))
        if not os.path.exists(mask_path):
            raise FileNotFoundError(mask_path)

        outputs = entry.get("outputs", {})
        new_mask_path = outputs.get("mask_new", "")
        if not new_mask_path or not os.path.exists(new_mask_path):
            raise FileNotFoundError(new_mask_path if new_mask_path else f"(missing outputs.mask_new) for {comp_name}")

        # pre mask (deletion): dilated old mask, at view resolution
        old_mask01 = _imread_mask01(mask_path)
        old_mask01 = cv2.resize(old_mask01, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        pre01 = _dilate_mask(old_mask01, remove_radius_px)

        # post mask (paste-back): dilated new mask, at view resolution
        post01 = _imread_mask01(new_mask_path)
        post01 = cv2.resize(post01, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        post01 = _dilate_mask(post01, remove_radius_px)

        # NEW: accumulate after masks
        sum_after01 = np.maximum(sum_after01, post01)          # union
        sum_after_count += post01.astype(np.uint16)            # count

        # diff: XOR
        diff01 = (pre01 ^ post01).astype(np.uint8)

        # expand diff "both ways"
        diff01 = _thicken_both_ways(diff01, radius_px=diff_thicken_px)

        diff_sum01 = np.maximum(diff_sum01, diff01)

        cv2.imwrite(os.path.join(out_dir, f"{comp_name}_pre_mask.png"), _mask01_to_png(pre01))
        cv2.imwrite(os.path.join(out_dir, f"{comp_name}_post_mask.png"), _mask01_to_png(post01))
        cv2.imwrite(os.path.join(out_dir, f"{comp_name}_diff_mask.png"), _mask01_to_png(diff01))

    # Save aggregates
    diff_sum_path = os.path.join(out_dir, "diff_sum_mask.png")
    sum_after_mask_path = os.path.join(out_dir, "sum_after_mask.png")  # binary union
    sum_after_count_npy_path = os.path.join(out_dir, "sum_after_count.npy")
    sum_after_count_png_path = os.path.join(out_dir, "sum_after_count.png")
    combined_path = os.path.join(out_dir, "sum_after_plus_diff_sum_mask.png")

    cv2.imwrite(diff_sum_path, _mask01_to_png(diff_sum01))
    cv2.imwrite(sum_after_mask_path, _mask01_to_png(sum_after01))

    np.save(sum_after_count_npy_path, sum_after_count)
    cv2.imwrite(sum_after_count_png_path, _count_to_vis_png(sum_after_count))

    combined01 = np.maximum(sum_after01, diff_sum01)
    cv2.imwrite(combined_path, _mask01_to_png(combined01))

    with open(os.path.join(out_dir, "_index.txt"), "w") as f:
        f.write(f"view_name: {view_name}\n")
        f.write(f"mask_warps_json: {mask_warps_json}\n")
        f.write(f"view_image: {view_img_path}\n")
        f.write(f"remove_radius_px(dilation): {remove_radius_px}\n")
        f.write(f"diff_thicken_px(both-ways): {diff_thicken_px}\n\n")

        f.write("Saved files per label:\n")
        for comp_name, _ in edited:
            f.write(f"- {comp_name}\n")
            f.write(f"  pre : {comp_name}_pre_mask.png\n")
            f.write(f"  post: {comp_name}_post_mask.png\n")
            f.write(f"  diff: {comp_name}_diff_mask.png\n")

        f.write("\nAggregate:\n")
        f.write("  diff_sum_mask.png\n")
        f.write("  sum_after_mask.png\n")
        f.write("  sum_after_count.npy\n")
        f.write("  sum_after_count.png\n")
        f.write("  sum_after_plus_diff_sum_mask.png\n")

    print(f"saved fix outputs to: {out_dir}")


def main():
    out_root = os.path.join("sketch", "final_outputs")

    warp_paths = sorted(glob.glob(os.path.join("sketch", "back_project_masks", "view_*", "mask_warps.json")))
    if not warp_paths:
        raise FileNotFoundError("No mask_warps.json found under sketch/back_project_masks/view_*/mask_warps.json")

    for p in warp_paths:
        process_view(p, out_root=out_root, remove_radius_px=5, diff_thicken_px=8)


if __name__ == "__main__":
    main()
