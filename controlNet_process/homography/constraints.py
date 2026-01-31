#!/usr/bin/env python3
# homography/constraints.py
#
# Compute close-boundary relations (within tolerance) between segmentation masks,
# sample sparse anchor points along those close boundaries, and save visualizations + JSON.
#
# Key fix:
#   - anchors are saved BOTH as absolute (image coords) AND as relative-to-mask-centroid offsets.
#   - Use *_dxy in downstream paste_back so anchors move with the mask.

import os
import json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEG_DIR = os.path.join(ROOT, "sketch", "segmentation_original_image")
VIEWS_DIR = os.path.join(ROOT, "sketch", "views")
OUT_ROOT = os.path.join(ROOT, "sketch", "back_project_masks")

NUM_VIEWS = 6
TOLERANCE_PIXELS = 10

# Anchor sampling controls
MIN_ANCHOR_SPACING_PX = 10
MAX_ANCHORS_PER_SIDE_PER_PAIR = 50

# Visualization controls
ANCHOR_DRAW_RADIUS = 2
ANCHOR_COLOR_L1 = (255, 0, 0)     # BGR: blue
ANCHOR_COLOR_L2 = (0, 0, 255)     # BGR: red
ANCHOR_COLOR_ALL = (0, 255, 255)  # BGR: yellow


# ----------------------------
# Geometry helpers
# ----------------------------
def _mask_centroid_xy(mask_u8: np.ndarray):
    """Centroid in image coords [cx, cy]. Returns None if empty."""
    if mask_u8 is None:
        return None
    ys, xs = np.nonzero(mask_u8 > 0)
    if xs.size == 0:
        return None
    return [float(np.mean(xs)), float(np.mean(ys))]


def _to_dxy(points_xy: np.ndarray, centroid_xy) -> np.ndarray:
    """Convert absolute points (x,y) to offsets (dx,dy) wrt centroid."""
    if points_xy is None or points_xy.size == 0 or centroid_xy is None:
        return np.zeros((0, 2), dtype=np.float64)
    c = np.array(centroid_xy, dtype=np.float64)[None, :]
    return (points_xy.astype(np.float64) - c)


def _from_dxy(points_dxy: np.ndarray, centroid_xy) -> np.ndarray:
    """Convert offsets (dx,dy) to absolute points (x,y) wrt centroid."""
    if points_dxy is None or points_dxy.size == 0 or centroid_xy is None:
        return np.zeros((0, 2), dtype=np.float64)
    c = np.array(centroid_xy, dtype=np.float64)[None, :]
    return (points_dxy.astype(np.float64) + c)


# ----------------------------
# Anchor sampling helpers
# ----------------------------
def _sample_points_with_min_spacing(
    bin_mask: np.ndarray,
    min_spacing_px: int,
    max_points: int,
) -> np.ndarray:
    """
    Sample (x,y) points from nonzero pixels of bin_mask, enforcing a minimum spacing.
    Returns up to max_points points.
    """
    ys, xs = np.nonzero(bin_mask)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    pts = np.stack([xs, ys], axis=1).astype(np.int32)

    # shuffle for spatial diversity
    order = np.random.permutation(pts.shape[0])
    pts = pts[order]

    chosen = []
    r2 = float(min_spacing_px * min_spacing_px)

    for p in pts:
        if not chosen:
            chosen.append(p)
        else:
            d = np.asarray(chosen) - p[None, :]
            if np.all((d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) >= r2):
                chosen.append(p)

        if len(chosen) >= max_points:
            break

    return np.asarray(chosen, dtype=np.int32)


def _merge_unique_points(points_list, max_points=None) -> np.ndarray:
    """Combine multiple (K,2) arrays into a unique set; optionally subsample."""
    if not points_list:
        return np.zeros((0, 2), dtype=np.int32)

    pts = np.concatenate(points_list, axis=0)
    if pts.shape[0] == 0:
        return pts.astype(np.int32)

    pts_unique = np.unique(pts, axis=0)

    if max_points is not None and pts_unique.shape[0] > max_points:
        idx = np.random.choice(pts_unique.shape[0], size=max_points, replace=False)
        pts_unique = pts_unique[idx]

    return pts_unique.astype(np.int32)


def _draw_points(img_bgr: np.ndarray, pts_xy: np.ndarray, color_bgr, radius: int = 2):
    """Draw small filled circles for each (x,y)."""
    for x, y in pts_xy:
        cv2.circle(img_bgr, (int(x), int(y)), radius, color_bgr, thickness=-1)


# ----------------------------
# Find close boundaries (+ anchors)
# ----------------------------
def find_close_boundary_pixels_and_anchors(boundaries_dict, tolerance_pixels=10):
    """
    Find boundary pixels from different masks that are close to each other,
    and extract sparse anchors on each side of each close pair.

    Returns:
        close_pixels: binary mask union of all close boundary pixels
        pair_info: list of dicts:
            {
              label1, label2,
              num_close_pixels,
              anchors_label1_xy: (K1,2) absolute [x,y] on label1 boundary close to label2,
              anchors_label2_xy: (K2,2) absolute [x,y] on label2 boundary close to label1
            }
    """
    labels = list(boundaries_dict.keys())
    H, W = next(iter(boundaries_dict.values())).shape

    close_pixels = np.zeros((H, W), dtype=np.uint8)
    pair_info = []

    kernel_size = 2 * tolerance_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for i, label1 in enumerate(labels):
        for label2 in labels[i + 1:]:
            boundary1 = boundaries_dict[label1].astype(np.uint8)
            boundary2 = boundaries_dict[label2].astype(np.uint8)

            dilated1 = cv2.dilate(boundary1, kernel, iterations=1)
            close_on_2 = (dilated1 > 0) & (boundary2 > 0)

            dilated2 = cv2.dilate(boundary2, kernel, iterations=1)
            close_on_1 = (dilated2 > 0) & (boundary1 > 0)

            close_region_union = close_on_1 | close_on_2
            num_close = int(np.sum(close_region_union))

            if num_close > 0:
                close_pixels[close_region_union] = 255

                anchors1 = _sample_points_with_min_spacing(
                    close_on_1.astype(np.uint8),
                    min_spacing_px=MIN_ANCHOR_SPACING_PX,
                    max_points=MAX_ANCHORS_PER_SIDE_PER_PAIR,
                )
                anchors2 = _sample_points_with_min_spacing(
                    close_on_2.astype(np.uint8),
                    min_spacing_px=MIN_ANCHOR_SPACING_PX,
                    max_points=MAX_ANCHORS_PER_SIDE_PER_PAIR,
                )

                pair_info.append({
                    "label1": label1,
                    "label2": label2,
                    "num_close_pixels": num_close,
                    "anchors_label1_xy": anchors1,  # absolute
                    "anchors_label2_xy": anchors2,  # absolute
                })

    return close_pixels, pair_info


# ----------------------------
# Main processing
# ----------------------------
def main():
    for x in range(NUM_VIEWS):
        view_name = f"view_{x}"

        seg_folder = os.path.join(SEG_DIR, view_name)
        img_path = os.path.join(VIEWS_DIR, f"{view_name}.png")

        if not os.path.exists(img_path) or not os.path.exists(seg_folder):
            continue

        base = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if base is None:
            continue

        H_img, W_img = base.shape[:2]

        out_dir = os.path.join(OUT_ROOT, view_name, "constraints")
        os.makedirs(out_dir, exist_ok=True)

        anchor_pairs_dir = os.path.join(out_dir, "anchor_pairs")
        os.makedirs(anchor_pairs_dir, exist_ok=True)

        # Load masks
        original_masks = {}
        label_centroids_xy = {}
        for mask_file in sorted(os.listdir(seg_folder)):
            if not mask_file.endswith("_mask.png"):
                continue

            label = mask_file.replace("_mask.png", "")
            mask_path = os.path.join(seg_folder, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            if mask.shape[:2] != (H_img, W_img):
                mask = cv2.resize(mask, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

            mask = (mask > 0).astype(np.uint8) * 255
            original_masks[label] = mask

            c = _mask_centroid_xy(mask)
            label_centroids_xy[label] = c  # may be None if empty, but usually not

        # Boundaries
        kernel = np.ones((3, 3), np.uint8)
        boundaries = {}
        for label, mask in original_masks.items():
            binary = (mask > 0).astype(np.uint8)
            eroded = cv2.erode(binary, kernel, iterations=1)
            boundaries[label] = (binary - eroded).astype(np.uint8) * 255

        # Close pixels + anchors (absolute)
        close_pixels, pair_info = find_close_boundary_pixels_and_anchors(
            boundaries,
            tolerance_pixels=TOLERANCE_PIXELS,
        )
        total_close = int(np.sum(close_pixels > 0))

        # Vis 1: masks white, boundaries red, close boundaries green
        vis1 = np.zeros((H_img, W_img, 3), dtype=np.uint8)

        all_masks = np.zeros((H_img, W_img), dtype=np.uint8)
        for mask in original_masks.values():
            all_masks[mask > 0] = 255
        vis1[all_masks > 0] = (255, 255, 255)

        all_boundaries = np.zeros((H_img, W_img), dtype=np.uint8)
        for boundary in boundaries.values():
            all_boundaries[boundary > 0] = 255
        vis1[all_boundaries > 0] = (0, 0, 255)

        vis1[close_pixels > 0] = (0, 255, 0)
        cv2.imwrite(os.path.join(out_dir, "boundaries_with_close_highlighted.png"), vis1)

        # Vis 2: only close pixels
        vis2 = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        vis2[close_pixels > 0] = (0, 255, 0)
        cv2.imwrite(os.path.join(out_dir, "close_boundaries_only.png"), vis2)

        # Per-label anchor unions (absolute + dxy)
        label_to_anchor_points_xy = {lbl: [] for lbl in original_masks.keys()}

        close_pairs_json = []
        for p in pair_info:
            l1 = p["label1"]
            l2 = p["label2"]
            anchors1_xy = p["anchors_label1_xy"]
            anchors2_xy = p["anchors_label2_xy"]

            label_to_anchor_points_xy[l1].append(anchors1_xy)
            label_to_anchor_points_xy[l2].append(anchors2_xy)

            c1 = label_centroids_xy.get(l1)
            c2 = label_centroids_xy.get(l2)

            anchors1_dxy = _to_dxy(anchors1_xy, c1)
            anchors2_dxy = _to_dxy(anchors2_xy, c2)

            # Pair visualization (still uses ABS for drawing)
            pair_vis = np.zeros((H_img, W_img, 3), dtype=np.uint8)

            pair_vis[boundaries[l1] > 0] = (255, 0, 0)  # blue boundary
            pair_vis[boundaries[l2] > 0] = (0, 0, 255)  # red boundary

            # close region highlight (green)
            kernel_size = 2 * TOLERANCE_PIXELS + 1
            kernel_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dil1 = cv2.dilate((boundaries[l1] > 0).astype(np.uint8), kernel_tol, iterations=1)
            dil2 = cv2.dilate((boundaries[l2] > 0).astype(np.uint8), kernel_tol, iterations=1)
            close_on_2 = (dil1 > 0) & (boundaries[l2] > 0)
            close_on_1 = (dil2 > 0) & (boundaries[l1] > 0)
            pair_vis[close_on_1 | close_on_2] = (0, 255, 0)

            _draw_points(pair_vis, anchors1_xy, ANCHOR_COLOR_L1, radius=ANCHOR_DRAW_RADIUS)
            _draw_points(pair_vis, anchors2_xy, ANCHOR_COLOR_L2, radius=ANCHOR_DRAW_RADIUS)

            pair_path = os.path.join(anchor_pairs_dir, f"{l1}__{l2}.png")
            cv2.imwrite(pair_path, pair_vis)

            close_pairs_json.append({
                "label1": l1,
                "label2": l2,
                "num_close_pixels": int(p["num_close_pixels"]),

                # ABSOLUTE (image coords) - kept for backwards compatibility
                "anchors_label1_xy": anchors1_xy.astype(int).tolist(),
                "anchors_label2_xy": anchors2_xy.astype(int).tolist(),

                # RELATIVE (mask-local offsets from centroid) - use these in paste_back
                "anchors_label1_dxy": anchors1_dxy.astype(float).tolist(),
                "anchors_label2_dxy": anchors2_dxy.astype(float).tolist(),

                "pair_anchor_vis": os.path.relpath(pair_path, out_dir),
            })

        # Per-label merged anchors
        label_anchor_points_xy = {}
        label_anchor_points_dxy = {}

        for lbl, pts_list in label_to_anchor_points_xy.items():
            merged_xy = _merge_unique_points(pts_list, max_points=2000)
            label_anchor_points_xy[lbl] = merged_xy.astype(int).tolist()

            c = label_centroids_xy.get(lbl)
            merged_dxy = _to_dxy(merged_xy, c)
            label_anchor_points_dxy[lbl] = merged_dxy.astype(float).tolist()

        # Vis 3: all anchors (yellow) over vis1 (ABS)
        all_anchor_vis = vis1.copy()
        all_pts_xy = _merge_unique_points(
            [np.array(v, dtype=np.int32) for v in label_anchor_points_xy.values() if len(v) > 0],
            max_points=5000,
        )
        _draw_points(all_anchor_vis, all_pts_xy, ANCHOR_COLOR_ALL, radius=ANCHOR_DRAW_RADIUS)
        cv2.imwrite(os.path.join(out_dir, "anchor_points_all.png"), all_anchor_vis)

        # JSON
        summary = {
            "view": view_name,
            "tolerance_pixels": int(TOLERANCE_PIXELS),
            "min_anchor_spacing_px": int(MIN_ANCHOR_SPACING_PX),
            "max_anchors_per_side_per_pair": int(MAX_ANCHORS_PER_SIDE_PER_PAIR),

            # new: label reference points for reconstructing ABS from DXY later
            "label_centroids_xy": label_centroids_xy,  # {label: [cx,cy] or None}

            "total_close_boundary_pixels": int(total_close),
            "num_pairs_with_close_boundaries": int(len(close_pairs_json)),
            "close_pairs": close_pairs_json,

            # ABS + DXY versions
            "label_anchor_points_xy": label_anchor_points_xy,
            "label_anchor_points_dxy": label_anchor_points_dxy,

            "outputs": {
                "boundaries_with_close_highlighted": "boundaries_with_close_highlighted.png",
                "close_boundaries_only": "close_boundaries_only.png",
                "anchor_points_all": "anchor_points_all.png",
                "anchor_pairs_dir": "anchor_pairs/",
            },
        }

        summary_path = os.path.join(out_dir, "close_boundaries_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
