#!/usr/bin/env python3
# 12_z_2d_homography.py
#
# Part 2: Homography Computation & Mask Warping
# - Reads 2D bounding box projections from part 1
# - Computes homographies from before/after 2D correspondences
# - Warps masks using homographies
# - Creates visualizations and overlays
#
# Inputs:
#   sketch/back_project_masks/view_{x}/3d_project/obb_2d_projections.json
#
# Outputs:
#   sketch/back_project_masks/view_{x}/{label}_mask_new.png
#   sketch/back_project_masks/view_{x}/{label}_origmask_overlay.png
#   sketch/back_project_masks/view_{x}/{label}_newmask_overlay.png
#   sketch/back_project_masks/view_{x}/all_new_masks_overlay.png
#   sketch/back_project_masks/view_{x}/mask_warps.json

import os
import json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))

VIEWS_DIR  = os.path.join(ROOT, "sketch", "views")
SEG_DIR    = os.path.join(ROOT, "sketch", "segmentation_original_image")
OUT_ROOT   = os.path.join(ROOT, "sketch", "back_project_masks")

NUM_VIEWS = 6

# Overlay style
ALPHA_FILL = 0.35
CONTOUR_THICK = 2

# Mask visualization colors (BGR)
ORIG_MASK_COLOR = (255, 0, 0)  # BLUE
NEW_MASK_COLOR  = (0, 0, 255)  # RED


# ----------------------------
# Visual + warp utilities
# ----------------------------
def color_for_label(label: str):
    h = (abs(hash(label)) % 180)
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def overlay_mask_on_image(img_bgr, mask_u8, color_bgr, alpha=0.35, contour_thick=2):
    out = img_bgr.copy()
    if mask_u8 is None:
        return out

    binary = mask_u8 > 0
    if not np.any(binary):
        return out

    fill = np.zeros_like(out)
    fill[:] = color_bgr
    out[binary] = cv2.addWeighted(out[binary], 1.0 - alpha, fill[binary], alpha, 0)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color_bgr, contour_thick)

    return out


def compute_homography_from_2d_points(p0, p1, valid):
    """
    Compute homography from before (p0) to after (p1) using valid correspondences.
    
    p0: (N,2) array of before 2D points
    p1: (N,2) array of after 2D points
    valid: (N,) bool array indicating valid correspondences
    
    Returns:
        Hmat: 3x3 homography matrix
        status_info: dict with status information
    """
    idx = np.where(valid)[0]

    if idx.size < 4:
        return np.eye(3, dtype=np.float64), {
            "status": "identity_fallback",
            "valid_pairs": int(idx.size),
        }

    src = np.array(p0)[idx].astype(np.float64)
    dst = np.array(p1)[idx].astype(np.float64)

    Hmat, _ = cv2.findHomography(src, dst, method=0)
    if Hmat is None:
        return np.eye(3, dtype=np.float64), {
            "status": "findHomography_failed_identity",
            "valid_pairs": int(idx.size),
        }

    return Hmat.astype(np.float64), {
        "status": "ok",
        "valid_pairs": int(idx.size),
    }


def warp_mask(mask_u8, Hmat, out_h, out_w):
    warped = cv2.warpPerspective(
        mask_u8,
        Hmat,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    warped = (warped > 0).astype(np.uint8) * 255
    return warped


# ----------------------------
# Main
# ----------------------------
def main():
    for x in range(NUM_VIEWS):
        view_name = f"view_{x}"
        img_path = os.path.join(VIEWS_DIR, f"{view_name}.png")
        seg_folder = os.path.join(SEG_DIR, view_name)
        proj_dir = os.path.join(OUT_ROOT, view_name, "3d_project")
        proj_json_path = os.path.join(proj_dir, "obb_2d_projections.json")

        if not os.path.exists(img_path):
            print(f"[skip] missing image: {img_path}")
            continue
        if not os.path.exists(seg_folder):
            print(f"[skip] missing seg folder: {seg_folder}")
            continue
        if not os.path.exists(proj_json_path):
            print(f"[skip] missing 2D projections: {proj_json_path}")
            continue

        base = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if base is None:
            print(f"[skip] failed to read image: {img_path}")
            continue

        H_img, W_img = base.shape[:2]

        # Load 2D projection data
        with open(proj_json_path, "r") as f:
            proj_data = json.load(f)

        out_dir = os.path.join(OUT_ROOT, view_name)
        os.makedirs(out_dir, exist_ok=True)

        all_new_overlay = base.copy()

        warp_json = {
            "view": view_name,
            "image": os.path.relpath(img_path, ROOT),
            "H": int(H_img),
            "W": int(W_img),
            "source_2d_projections": os.path.relpath(proj_json_path, ROOT),
            "labels": {}
        }

        for label, proj_info in proj_data["labels"].items():
            # Extract 2D points
            p0 = np.array(proj_info["obb_before_2d"], dtype=np.float64)
            p1 = np.array(proj_info["obb_after_2d"], dtype=np.float64)
            valid = np.array(proj_info["valid_pairs"], dtype=bool)

            # Check if mask exists
            mask_path = os.path.join(seg_folder, f"{label}_mask.png")
            has_mask = os.path.exists(mask_path)

            label_entry = {
                "has_mask": bool(has_mask),
                "mask_path": os.path.relpath(mask_path, ROOT) if has_mask else None,
                "obb_before_after_overlay": proj_info["overlay_image"],
                "homography_before_to_after": None,
                "homography_status": None,
                "valid_pairs": int(proj_info["num_valid_pairs"]),
                "outputs": {}
            }

            if not has_mask:
                warp_json["labels"][label] = label_entry
                print(f"[skip] {view_name} {label}: no mask found")
                continue

            mask0 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask0 is None:
                warp_json["labels"][label] = label_entry
                print(f"[skip] {view_name} {label}: failed to read mask")
                continue
            
            if mask0.shape[:2] != (H_img, W_img):
                mask0 = cv2.resize(mask0, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

            # Compute homography
            Hmat, dbg = compute_homography_from_2d_points(p0, p1, valid)
            
            # Warp mask
            mask1 = warp_mask(mask0, Hmat, H_img, W_img)

            # Save new mask
            out_mask_path = os.path.join(out_dir, f"{label}_mask_new.png")
            cv2.imwrite(out_mask_path, mask1)

            # Create overlays (original mask = BLUE, new mask = RED)
            orig_overlay = overlay_mask_on_image(
                base, mask0, ORIG_MASK_COLOR, alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK
            )
            new_overlay = overlay_mask_on_image(
                base, mask1, NEW_MASK_COLOR, alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK
            )

            out_orig_overlay = os.path.join(out_dir, f"{label}_origmask_overlay.png")
            out_new_overlay  = os.path.join(out_dir, f"{label}_newmask_overlay.png")

            cv2.imwrite(out_orig_overlay, orig_overlay)
            cv2.imwrite(out_new_overlay, new_overlay)

            # Aggregate overlay (per-label color)
            col = color_for_label(label)
            all_new_overlay = overlay_mask_on_image(all_new_overlay, mask1, col, alpha=ALPHA_FILL, contour_thick=1)

            # Store results
            label_entry["homography_before_to_after"] = Hmat.tolist()
            label_entry["homography_status"] = dbg.get("status")
            label_entry["valid_pairs"] = int(dbg.get("valid_pairs", 0))
            label_entry["outputs"] = {
                "mask_new": os.path.relpath(out_mask_path, ROOT),
                "origmask_overlay": os.path.relpath(out_orig_overlay, ROOT),
                "newmask_overlay": os.path.relpath(out_new_overlay, ROOT),
            }

            warp_json["labels"][label] = label_entry

            print(f"[ok] {view_name} {label}: mask warped | H={dbg['status']} pairs={dbg.get('valid_pairs')}")

        # Save aggregate overlay
        out_all = os.path.join(out_dir, "all_new_masks_overlay.png")
        cv2.imwrite(out_all, all_new_overlay)
        print(f"[ok] {view_name}: wrote {out_all}")

        # Save warp JSON
        warp_json["all_new_masks_overlay"] = os.path.relpath(out_all, ROOT)
        warp_json_path = os.path.join(out_dir, "mask_warps.json")
        with open(warp_json_path, "w") as f:
            json.dump(warp_json, f, indent=2)
        print(f"[ok] {view_name}: wrote {warp_json_path}")

    print("[done] All masks warped and visualizations created.")


if __name__ == "__main__":
    main()