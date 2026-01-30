#!/usr/bin/env python3
# homography/homography.py
#
# Homography Computation & Mask Warping
# - Reads 2D bounding box projections from part 1
# - Computes homographies from before/after 2D correspondences
# - Warps masks using homographies
# - Creates visualizations and overlays
#
# Inputs:
#   sketch/back_project_masks/view_{x}/3d_project/obb_2d_projections.json
#   sketch/segmentation_original_image/view_{x}/{label}_mask.png
#   sketch/views/view_{x}.png
#
# Outputs:
#   sketch/back_project_masks/view_{x}/homography/{label}_mask_warped.png
#   sketch/back_project_masks/view_{x}/homography/{label}_origmask_overlay.png
#   sketch/back_project_masks/view_{x}/homography/{label}_warpedmask_overlay.png
#   sketch/back_project_masks/view_{x}/homography/all_warped_masks_overlay.png
#   sketch/back_project_masks/view_{x}/homography/homography_results.json

import os
import json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIEWS_DIR  = os.path.join(ROOT, "sketch", "views")
SEG_DIR    = os.path.join(ROOT, "sketch", "segmentation_original_image")
OUT_ROOT   = os.path.join(ROOT, "sketch", "back_project_masks")

NUM_VIEWS = 6

# Overlay style
ALPHA_FILL = 0.35
CONTOUR_THICK = 2

# Mask visualization colors (BGR)
ORIG_MASK_COLOR = (255, 0, 0)  # BLUE
WARP_MASK_COLOR = (0, 0, 255)  # RED


# ----------------------------
# Visual utilities
# ----------------------------
def color_for_label(label: str):
    """Generate a unique color for each label based on hash."""
    h = (abs(hash(label)) % 180)
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def overlay_mask_on_image(img_bgr, mask_u8, color_bgr, alpha=0.35, contour_thick=2):
    """Overlay a mask on an image with transparency and contour."""
    out = img_bgr.copy()
    if mask_u8 is None:
        return out

    binary = mask_u8 > 0
    if not np.any(binary):
        return out

    # Fill with transparency
    fill = np.zeros_like(out)
    fill[:] = color_bgr
    out[binary] = cv2.addWeighted(out[binary], 1.0 - alpha, fill[binary], alpha, 0)

    # Draw contours
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color_bgr, contour_thick)

    return out


# ----------------------------
# Homography computation
# ----------------------------
def compute_homography_from_2d_points(p0, p1, valid):
    """
    Compute homography from before (p0) to after (p1) using valid correspondences.
    
    Args:
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
            "reason": "insufficient_correspondences",
            "valid_pairs": int(idx.size),
        }

    src = np.array(p0)[idx].astype(np.float64)
    dst = np.array(p1)[idx].astype(np.float64)

    Hmat, _ = cv2.findHomography(src, dst, method=0)
    if Hmat is None:
        return np.eye(3, dtype=np.float64), {
            "status": "identity_fallback",
            "reason": "findHomography_failed",
            "valid_pairs": int(idx.size),
        }

    return Hmat.astype(np.float64), {
        "status": "ok",
        "valid_pairs": int(idx.size),
    }


def warp_mask(mask_u8, Hmat, out_h, out_w):
    """Warp a mask using a homography matrix."""
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


def compute_mask_centroid(mask_u8):
    """Compute the centroid of a mask."""
    if mask_u8 is None:
        return None
    
    binary = mask_u8 > 0
    if not np.any(binary):
        return None
    
    ys, xs = np.where(binary)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    return [cx, cy]


# ----------------------------
# Main processing
# ----------------------------
def main():
    for x in range(NUM_VIEWS):
        view_name = f"view_{x}"
        img_path = os.path.join(VIEWS_DIR, f"{view_name}.png")
        seg_folder = os.path.join(SEG_DIR, view_name)
        proj_dir = os.path.join(OUT_ROOT, view_name, "3d_project")
        proj_json_path = os.path.join(proj_dir, "obb_2d_projections.json")

        # Check inputs
        if not os.path.exists(img_path):
            print(f"[skip] {view_name}: missing image: {img_path}")
            continue
        if not os.path.exists(seg_folder):
            print(f"[skip] {view_name}: missing seg folder: {seg_folder}")
            continue
        if not os.path.exists(proj_json_path):
            print(f"[skip] {view_name}: missing 2D projections: {proj_json_path}")
            continue

        # Load base image
        base = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if base is None:
            print(f"[skip] {view_name}: failed to read image: {img_path}")
            continue

        H_img, W_img = base.shape[:2]

        # Load 2D projection data
        with open(proj_json_path, "r") as f:
            proj_data = json.load(f)

        # Create output directory
        out_dir = os.path.join(OUT_ROOT, view_name, "homography")
        os.makedirs(out_dir, exist_ok=True)

        # Initialize aggregate overlay
        all_warped_overlay = base.copy()

        # Initialize results JSON
        results_json = {
            "view": view_name,
            "image": os.path.relpath(img_path, ROOT),
            "H": int(H_img),
            "W": int(W_img),
            "source_2d_projections": os.path.relpath(proj_json_path, ROOT),
            "labels": {}
        }

        # Process each label
        for label, proj_info in proj_data["labels"].items():
            # Extract 2D points
            p0 = np.array(proj_info["obb_before_2d"], dtype=np.float64)
            p1 = np.array(proj_info["obb_after_2d"], dtype=np.float64)
            valid = np.array(proj_info["valid_pairs"], dtype=bool)

            # Check if mask exists
            mask_path = os.path.join(seg_folder, f"{label}_mask.png")
            has_mask = os.path.exists(mask_path)

            # Initialize label entry
            label_entry = {
                "has_mask": bool(has_mask),
                "mask_path": os.path.relpath(mask_path, ROOT) if has_mask else None,
                "obb_before_after_overlay": proj_info["overlay_image"],
                "homography_before_to_after": None,
                "homography_status": None,
                "valid_pairs": int(proj_info["num_valid_pairs"]),
                "original_centroid": None,
                "warped_centroid": None,
                "outputs": {}
            }

            if not has_mask:
                results_json["labels"][label] = label_entry
                continue

            # Load mask
            mask0 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask0 is None:
                results_json["labels"][label] = label_entry
                continue
            
            # Resize mask if needed
            if mask0.shape[:2] != (H_img, W_img):
                mask0 = cv2.resize(mask0, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

            # Compute original centroid
            orig_centroid = compute_mask_centroid(mask0)

            # Compute homography
            Hmat, dbg = compute_homography_from_2d_points(p0, p1, valid)
            
            # Warp mask
            mask_warped = warp_mask(mask0, Hmat, H_img, W_img)

            # Compute warped centroid
            warped_centroid = compute_mask_centroid(mask_warped)

            # Save warped mask
            out_mask_path = os.path.join(out_dir, f"{label}_mask_warped.png")
            cv2.imwrite(out_mask_path, mask_warped)

            # Create overlays (original mask = BLUE, warped mask = RED)
            orig_overlay = overlay_mask_on_image(
                base, mask0, ORIG_MASK_COLOR, alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK
            )
            warped_overlay = overlay_mask_on_image(
                base, mask_warped, WARP_MASK_COLOR, alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK
            )

            out_orig_overlay = os.path.join(out_dir, f"{label}_origmask_overlay.png")
            out_warped_overlay = os.path.join(out_dir, f"{label}_warpedmask_overlay.png")

            cv2.imwrite(out_orig_overlay, orig_overlay)
            cv2.imwrite(out_warped_overlay, warped_overlay)

            # Aggregate overlay (per-label color)
            col = color_for_label(label)
            all_warped_overlay = overlay_mask_on_image(
                all_warped_overlay, mask_warped, col, alpha=ALPHA_FILL, contour_thick=1
            )

            # Store results
            label_entry["homography_before_to_after"] = Hmat.tolist()
            label_entry["homography_status"] = dbg.get("status")
            label_entry["valid_pairs"] = int(dbg.get("valid_pairs", 0))
            label_entry["original_centroid"] = orig_centroid
            label_entry["warped_centroid"] = warped_centroid
            label_entry["outputs"] = {
                "mask_warped": os.path.relpath(out_mask_path, ROOT),
                "origmask_overlay": os.path.relpath(out_orig_overlay, ROOT),
                "warpedmask_overlay": os.path.relpath(out_warped_overlay, ROOT),
            }

            results_json["labels"][label] = label_entry

        # Save aggregate overlay
        out_all = os.path.join(out_dir, "all_warped_masks_overlay.png")
        cv2.imwrite(out_all, all_warped_overlay)

        # Save results JSON
        results_json["all_warped_masks_overlay"] = os.path.relpath(out_all, ROOT)
        results_json_path = os.path.join(out_dir, "homography_results.json")
        with open(results_json_path, "w") as f:
            json.dump(results_json, f, indent=2)

    print("[done] Homography computation complete.")


if __name__ == "__main__":
    main()