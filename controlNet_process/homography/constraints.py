#!/usr/bin/env python3
# homography/constraints.py
#
# Compute close-boundary relations (within tolerance) between segmentation masks,
# sample sparse anchor points along those close boundaries, and save visualizations + JSON.
#
# NOW ALSO SAVES PAIRED ANCHORS (CORRESPONDENCES) SO paste_back DOES NOT GUESS PAIRING.

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
ANCHOR_COLOR_ALL = (0, 255, 255)  # BGR: yellow (for combined vis)


# ----------------------------
# Anchor sampling helpers
# ----------------------------
def _sample_points_with_min_spacing(
    bin_mask: np.ndarray,
    min_spacing_px: int,
    max_points: int,
) -> np.ndarray:
    ys, xs = np.nonzero(bin_mask)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    pts = np.stack([xs, ys], axis=1).astype(np.int32)

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
    for x, y in pts_xy:
        cv2.circle(img_bgr, (int(x), int(y)), radius, color_bgr, thickness=-1)


def _nearest_boundary_pixel_in_window(
    boundary_u8: np.ndarray,
    x: int,
    y: int,
    tol: int,
) -> np.ndarray:
    """
    Find nearest boundary pixel in boundary_u8 within a (2*tol+1)x(2*tol+1) window around (x,y).
    Returns [xn, yn] int32 or None if no boundary pixels in window.
    """
    H, W = boundary_u8.shape[:2]
    x0 = max(0, x - tol)
    x1 = min(W - 1, x + tol)
    y0 = max(0, y - tol)
    y1 = min(H - 1, y + tol)

    patch = boundary_u8[y0:y1 + 1, x0:x1 + 1]
    ys, xs = np.nonzero(patch > 0)
    if xs.size == 0:
        return None

    # convert to absolute coords
    xs_abs = xs + x0
    ys_abs = ys + y0

    dx = xs_abs.astype(np.float64) - float(x)
    dy = ys_abs.astype(np.float64) - float(y)
    d2 = dx * dx + dy * dy
    k = int(np.argmin(d2))
    return np.array([int(xs_abs[k]), int(ys_abs[k])], dtype=np.int32)


def _pair_anchors_by_local_nearest(
    anchors_src_xy: np.ndarray,
    boundary_tgt_u8: np.ndarray,
    tol: int,
) -> np.ndarray:
    """
    For each anchor in anchors_src_xy (on src boundary), find nearest pixel on target boundary
    within tolerance window. Returns (K,2,2): [[[x_src,y_src],[x_tgt,y_tgt]], ...]
    """
    pairs = []
    for (x, y) in anchors_src_xy:
        nn = _nearest_boundary_pixel_in_window(boundary_tgt_u8, int(x), int(y), tol)
        if nn is None:
            continue
        pairs.append([[int(x), int(y)], [int(nn[0]), int(nn[1])]])
    return np.array(pairs, dtype=np.int32) if len(pairs) > 0 else np.zeros((0, 2, 2), dtype=np.int32)


# ----------------------------
# Find close boundaries (+ anchors)
# ----------------------------
def find_close_boundary_pixels_and_anchors(boundaries_dict, tolerance_pixels=10):
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
            close_on_2 = (dilated1 > 0) & (boundary2 > 0)   # pixels on boundary2 close to boundary1

            dilated2 = cv2.dilate(boundary2, kernel, iterations=1)
            close_on_1 = (dilated2 > 0) & (boundary1 > 0)   # pixels on boundary1 close to boundary2

            close_region_union = close_on_1 | close_on_2
            num_close = int(np.sum(close_region_union))

            if num_close <= 0:
                continue

            close_pixels[close_region_union] = 255

            # Sample anchors on each side (still useful for visualization + per-label unions)
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

            # NEW: build *paired* anchors using local nearest mapping (no guessing later)
            # Pairing direction is explicitly label1 -> label2 and label2 -> label1
            paired_12 = _pair_anchors_by_local_nearest(anchors1, boundary2, tolerance_pixels)  # [ [l1],[l2] ]
            paired_21 = _pair_anchors_by_local_nearest(anchors2, boundary1, tolerance_pixels)  # [ [l2],[l1] ]

            pair_info.append({
                "label1": label1,
                "label2": label2,
                "num_close_pixels": num_close,
                "anchors_label1_xy": anchors1,
                "anchors_label2_xy": anchors2,
                "paired_anchors_l1_l2_xy": paired_12,  # shape (K,2,2): [[[x1,y1],[x2,y2]],...]
                "paired_anchors_l2_l1_xy": paired_21,  # shape (K,2,2): [[[x2,y2],[x1,y1]],...]
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
            original_masks[label] = mask

        # Boundaries
        kernel = np.ones((3, 3), np.uint8)
        boundaries = {}
        for label, mask in original_masks.items():
            binary = (mask > 0).astype(np.uint8)
            eroded = cv2.erode(binary, kernel, iterations=1)
            boundaries[label] = binary - eroded

        # Close pixels + anchors (+ paired anchors)
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

        # Per-label anchor unions
        label_to_anchor_points = {lbl: [] for lbl in original_masks.keys()}

        close_pairs_json = []
        for p in pair_info:
            l1 = p["label1"]
            l2 = p["label2"]
            anchors1 = p["anchors_label1_xy"]
            anchors2 = p["anchors_label2_xy"]

            label_to_anchor_points[l1].append(anchors1)
            label_to_anchor_points[l2].append(anchors2)

            # Pair visualization (+ show pairing lines for l1->l2 correspondences)
            pair_vis = np.zeros((H_img, W_img, 3), dtype=np.uint8)

            pair_vis[boundaries[l1] > 0] = (255, 0, 0)  # blue boundary
            pair_vis[boundaries[l2] > 0] = (0, 0, 255)  # red boundary

            kernel_size = 2 * TOLERANCE_PIXELS + 1
            kernel_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dil1 = cv2.dilate(boundaries[l1].astype(np.uint8), kernel_tol, iterations=1)
            dil2 = cv2.dilate(boundaries[l2].astype(np.uint8), kernel_tol, iterations=1)
            close_on_2 = (dil1 > 0) & (boundaries[l2] > 0)
            close_on_1 = (dil2 > 0) & (boundaries[l1] > 0)
            pair_vis[close_on_1 | close_on_2] = (0, 255, 0)

            _draw_points(pair_vis, anchors1, ANCHOR_COLOR_L1, radius=ANCHOR_DRAW_RADIUS)
            _draw_points(pair_vis, anchors2, ANCHOR_COLOR_L2, radius=ANCHOR_DRAW_RADIUS)

            # draw lines for paired_anchors_l1_l2_xy (cyan)
            paired_12 = p["paired_anchors_l1_l2_xy"]
            if paired_12.shape[0] > 0:
                for a, b in paired_12:
                    ax, ay = int(a[0]), int(a[1])
                    bx, by = int(b[0]), int(b[1])
                    cv2.line(pair_vis, (ax, ay), (bx, by), (255, 255, 0), 1)

            pair_path = os.path.join(anchor_pairs_dir, f"{l1}__{l2}.png")
            cv2.imwrite(pair_path, pair_vis)

            close_pairs_json.append({
                "label1": l1,
                "label2": l2,
                "num_close_pixels": int(p["num_close_pixels"]),
                "anchors_label1_xy": anchors1.astype(int).tolist(),
                "anchors_label2_xy": anchors2.astype(int).tolist(),
                "paired_anchors_l1_l2_xy": p["paired_anchors_l1_l2_xy"].astype(int).tolist(),
                "paired_anchors_l2_l1_xy": p["paired_anchors_l2_l1_xy"].astype(int).tolist(),
                "pair_anchor_vis": os.path.relpath(pair_path, out_dir),
            })

        label_anchors_json = {}
        for lbl, pts_list in label_to_anchor_points.items():
            merged = _merge_unique_points(pts_list, max_points=2000)
            label_anchors_json[lbl] = merged.astype(int).tolist()

        # Vis 3: all anchors (yellow) over vis1
        all_anchor_vis = vis1.copy()
        all_pts = _merge_unique_points(
            [np.array(v, dtype=np.int32) for v in label_anchors_json.values() if len(v) > 0],
            max_points=5000,
        )
        _draw_points(all_anchor_vis, all_pts, ANCHOR_COLOR_ALL, radius=ANCHOR_DRAW_RADIUS)
        cv2.imwrite(os.path.join(out_dir, "anchor_points_all.png"), all_anchor_vis)

        # JSON
        summary = {
            "view": view_name,
            "tolerance_pixels": int(TOLERANCE_PIXELS),
            "min_anchor_spacing_px": int(MIN_ANCHOR_SPACING_PX),
            "max_anchors_per_side_per_pair": int(MAX_ANCHORS_PER_SIDE_PER_PAIR),
            "total_close_boundary_pixels": int(total_close),
            "num_pairs_with_close_boundaries": int(len(close_pairs_json)),
            "close_pairs": close_pairs_json,
            "label_anchor_points_xy": label_anchors_json,
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
