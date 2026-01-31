#!/usr/bin/env python3
# 4_2_clean_up_mask.py
#
# Run from controlNet_process/:
#   (baseline_sketch) ... controlNet_process % python 4_2_clean_up_mask.py
#
# IMPORTANT CHANGE:
# - DO NOT use view-derived "overall shape" at all.
# - For masks copied from archive (or any masks), only:
#     (1) fill enclosed holes => pure white inside boundary
#     (2) optional boundary adjust (dilate/erode)
#     (3) enforce no overlaps
#     (4) optional unlabeled assignment can be disabled (default disabled here)

import os
import json
import shutil
import numpy as np
import cv2

ROOT = os.getcwd()

SEG_DIR     = os.path.join(ROOT, "sketch", "segmentation_original_image")
SEG_ARCHIVE = os.path.join(ROOT, "sketch", "segmentation_original_image_archive")
VIEWS_DIR   = os.path.join(ROOT, "sketch", "views")

NUM_VIEWS = 6

# Low-priority labels (lose on overlap). You said: "now, just put storage_0 there".
LOW_PRIORITY_LABELS = ["storage_0"]

# Boundary adjust:
#  0  => keep as-is
# +k  => dilate by k px  (include boundary)
# -k  => erode by k px   (decrease boundary)
BOUNDARY_ADJUST_PX = 0

# Unlabeled assignment (DISABLED by default since you said don't use view shape at all)
DO_ASSIGN_UNLABELED = False
ASSIGN_DIST_PX = 5.0

# Outputs in the same folder (leave True if you still want them)
SAVE_OVERLAYS_AND_CUTOUTS = True
ALPHA_SINGLE = 0.85
ALPHA_ALL    = 0.70
CUTOUT_EXPAND_PX = 2  # for {label}.png alpha dilation only


def _imread_gray(path: str):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def _imread_bgr(path: str):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _ensure_u8_binary(mask_u8: np.ndarray) -> np.ndarray:
    if mask_u8 is None:
        return None
    return ((mask_u8 > 0).astype(np.uint8) * 255)


def _seg_is_empty(seg_dir: str) -> bool:
    if not os.path.isdir(seg_dir):
        return True
    for dirpath, _, filenames in os.walk(seg_dir):
        for fn in filenames:
            if fn.endswith("_mask.png"):
                return False
    return True


def _copy_archive_if_needed():
    if not _seg_is_empty(SEG_DIR):
        return
    if not os.path.isdir(SEG_ARCHIVE):
        raise FileNotFoundError(f"segmentation_original_image is empty AND archive missing: {SEG_ARCHIVE}")

    os.makedirs(SEG_DIR, exist_ok=True)
    for root, dirs, files in os.walk(SEG_ARCHIVE):
        rel = os.path.relpath(root, SEG_ARCHIVE)
        dst_root = os.path.join(SEG_DIR, rel) if rel != "." else SEG_DIR
        os.makedirs(dst_root, exist_ok=True)
        for d in dirs:
            os.makedirs(os.path.join(dst_root, d), exist_ok=True)
        for fn in files:
            shutil.copy2(os.path.join(root, fn), os.path.join(dst_root, fn))

    print("[init] segmentation_original_image was empty; copied from archive")


def _load_masks(seg_folder: str, target_hw=None):
    masks = {}
    if not os.path.isdir(seg_folder):
        return masks
    for fn in sorted(os.listdir(seg_folder)):
        if not fn.endswith("_mask.png"):
            continue
        label = fn[:-len("_mask.png")]
        path = os.path.join(seg_folder, fn)
        m = _imread_gray(path)
        if m is None:
            continue
        if target_hw is not None and m.shape[:2] != target_hw:
            H, W = target_hw
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        masks[label] = _ensure_u8_binary(m)
    return masks


def _boundary_adjust(mask_u8: np.ndarray, delta_px: int) -> np.ndarray:
    if delta_px == 0:
        return mask_u8
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(delta_px) + 1, 2 * abs(delta_px) + 1))
    if delta_px > 0:
        return cv2.dilate(mask_u8, k, iterations=1)
    else:
        return cv2.erode(mask_u8, k, iterations=1)


def _fill_enclosed_holes(mask_u8: np.ndarray) -> np.ndarray:
    """
    Fill *holes* inside the mask while preserving outer boundary.
    This is what makes "pure white inside".
    """
    m = (mask_u8 > 0).astype(np.uint8) * 255
    if int(np.sum(m > 0)) == 0:
        return m

    # Invert mask: background becomes 255
    inv = cv2.bitwise_not(m)

    # Flood-fill from border to mark "true background"
    h, w = m.shape[:2]
    ff = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, seedPoint=(0, 0), newVal=0)

    # Holes are what remained in inv that were NOT connected to border
    holes = (ff > 0)
    filled = m.copy()
    filled[holes] = 255
    return filled


def _postprocess_masks_inplace(masks: dict, boundary_adjust_px: int):
    """
    For each mask:
      - fill holes (pure white inside)
      - optional boundary adjust
      - fill holes again (after morph ops)
    """
    for lbl in list(masks.keys()):
        m = masks[lbl]
        m = _fill_enclosed_holes(m)
        m = _boundary_adjust(m, boundary_adjust_px)
        m = _fill_enclosed_holes(m)
        masks[lbl] = _ensure_u8_binary(m)


def _enforce_no_overlaps_inplace(masks: dict, low_priority_labels: list):
    """
    No overlaps:
      - low priority always loses on overlap
      - otherwise cut lexicographically larger label
    """
    labels = sorted(list(masks.keys()))
    if not labels:
        return

    low = set(low_priority_labels)

    # pairwise resolve overlaps
    for i, a in enumerate(labels):
        a_bin = (masks[a] > 0)
        if not np.any(a_bin):
            continue
        for b in labels[i + 1:]:
            b_bin = (masks[b] > 0)
            if not np.any(b_bin):
                continue

            inter = a_bin & b_bin
            if not np.any(inter):
                continue

            if (a in low) and (b not in low):
                loser = a
            elif (b in low) and (a not in low):
                loser = b
            else:
                loser = b if b > a else a

            if loser == a:
                a_bin = a_bin & (~inter)
                masks[a] = (a_bin.astype(np.uint8) * 255)
            else:
                b_bin = b_bin & (~inter)
                masks[b] = (b_bin.astype(np.uint8) * 255)

    # After cutting, re-fill holes so interiors are solid again (but won't refill removed overlap if it's open)
    _postprocess_masks_inplace(masks, boundary_adjust_px=0)


def _distance_to_mask_foreground(mask_u8: np.ndarray) -> np.ndarray:
    inv = (mask_u8 == 0).astype(np.uint8)  # bg=1, fg=0
    return cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)


def _assign_unlabeled_to_nearest_inplace(masks: dict, dist_thresh: float):
    """
    If you really still want it WITHOUT using view shape:
    define unlabeled = (dilated union) - union
    (This only fills tiny gaps between components; it won't invent new shape.)
    """
    if not masks:
        return 0, 0

    labels = sorted(list(masks.keys()))
    H, W = next(iter(masks.values())).shape[:2]

    union = np.zeros((H, W), dtype=np.uint8)
    for m in masks.values():
        union = cv2.bitwise_or(union, (m > 0).astype(np.uint8) * 255)

    # "overall region" = slightly dilated union
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # ~5px radius
    overall = cv2.dilate(union, k, iterations=1)

    unlabeled = (overall > 0) & (union == 0)
    unlabeled_count = int(np.sum(unlabeled))
    if unlabeled_count == 0:
        return 0, 0

    min_dist = np.full((H, W), np.inf, dtype=np.float32)
    best_idx = np.full((H, W), -1, dtype=np.int32)

    for i, lbl in enumerate(labels):
        dist = _distance_to_mask_foreground(masks[lbl])
        better = dist < min_dist
        min_dist[better] = dist[better]
        best_idx[better] = i

    assignable = unlabeled & (min_dist < float(dist_thresh))
    assigned = int(np.sum(assignable))
    if assigned == 0:
        return unlabeled_count, 0

    for i, lbl in enumerate(labels):
        pick = assignable & (best_idx == i)
        if np.any(pick):
            masks[lbl][pick] = 255

    return unlabeled_count, assigned


def _label_color_bgr_vivid(label: str):
    # vivid HSV -> BGR
    h = 2166136261
    for c in label.encode("utf-8"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    hue = int(h % 180)
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _overlay_color(base_bgr: np.ndarray, mask_u8: np.ndarray, color_bgr, alpha: float):
    out = base_bgr.copy()
    m = (mask_u8 > 0)
    if not np.any(m):
        return out
    color_img = np.zeros_like(out, dtype=np.uint8)
    color_img[:, :] = np.array(color_bgr, dtype=np.uint8)
    out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + color_img[m].astype(np.float32) * alpha).astype(np.uint8)
    return out


def _dilate_mask(mask_u8: np.ndarray, r_px: int) -> np.ndarray:
    if r_px <= 0:
        return mask_u8
    ksz = 2 * r_px + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    return cv2.dilate(mask_u8, k, iterations=1)


def _cutout_rgba(view_bgr: np.ndarray, mask_u8: np.ndarray, expand_px: int) -> np.ndarray:
    H, W = mask_u8.shape[:2]
    if view_bgr.shape[:2] != (H, W):
        view_bgr = cv2.resize(view_bgr, (W, H), interpolation=cv2.INTER_AREA)
    alpha = _dilate_mask(mask_u8, expand_px)
    bgra = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def main():
    _copy_archive_if_needed()

    for x in range(NUM_VIEWS):
        view_name = f"view_{x}"
        seg_folder = os.path.join(SEG_DIR, view_name)
        view_path  = os.path.join(VIEWS_DIR, f"{view_name}.png")

        if not os.path.isdir(seg_folder):
            continue

        view_bgr = _imread_bgr(view_path) if os.path.exists(view_path) else None

        # Load masks
        target_hw = None
        if view_bgr is not None:
            target_hw = view_bgr.shape[:2]
        masks = _load_masks(seg_folder, target_hw=target_hw)

        if not masks:
            print(f"[{view_name}] no masks found")
            continue

        # 1) Make masks "pure white inside" + optional boundary adjust
        _postprocess_masks_inplace(masks, boundary_adjust_px=BOUNDARY_ADJUST_PX)

        # 2) Enforce no overlaps (low priority loses)
        _enforce_no_overlaps_inplace(masks, low_priority_labels=LOW_PRIORITY_LABELS)

        # 3) OPTIONAL: fill tiny unlabeled gaps near components (without using view shape)
        unlabeled_before = 0
        assigned = 0
        if DO_ASSIGN_UNLABELED:
            unlabeled_before, assigned = _assign_unlabeled_to_nearest_inplace(masks, dist_thresh=ASSIGN_DIST_PX)
            _enforce_no_overlaps_inplace(masks, low_priority_labels=LOW_PRIORITY_LABELS)

        # 4) Overwrite masks + (optional) artifacts in the same folder
        for lbl, m in masks.items():
            cv2.imwrite(os.path.join(seg_folder, f"{lbl}_mask.png"), m)

        if SAVE_OVERLAYS_AND_CUTOUTS and view_bgr is not None:
            # per-label overlay + cutout
            for lbl in sorted(masks.keys()):
                m = masks[lbl]
                overlay = _overlay_color(view_bgr, m, _label_color_bgr_vivid(lbl), alpha=ALPHA_SINGLE)
                cv2.imwrite(os.path.join(seg_folder, f"{lbl}_overlay.png"), overlay)

                cut = _cutout_rgba(view_bgr, m, expand_px=CUTOUT_EXPAND_PX)
                cv2.imwrite(os.path.join(seg_folder, f"{lbl}.png"), cut)

            # all masks overlay
            all_overlay = view_bgr.copy()
            for lbl in sorted(masks.keys()):
                all_overlay = _overlay_color(all_overlay, masks[lbl], _label_color_bgr_vivid(lbl), alpha=ALPHA_ALL)
            cv2.imwrite(os.path.join(seg_folder, "all_masks_overlay.png"), all_overlay)

        # summary
        summary = {
            "view": view_name,
            "low_priority_labels": LOW_PRIORITY_LABELS,
            "boundary_adjust_px": int(BOUNDARY_ADJUST_PX),
            "do_assign_unlabeled": bool(DO_ASSIGN_UNLABELED),
            "assign_dist_px": float(ASSIGN_DIST_PX),
            "unlabeled_before": int(unlabeled_before),
            "assigned_pixels": int(assigned),
            "labels": sorted(list(masks.keys())),
            "notes": {
                "shape_clip": "DISABLED (no view-derived shape is used)",
                "mask_fill": "holes filled via flood-fill => pure white inside boundary",
            },
        }
        with open(os.path.join(seg_folder, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[{view_name}] wrote {len(masks)} masks")

    print("done.")


if __name__ == "__main__":
    main()
