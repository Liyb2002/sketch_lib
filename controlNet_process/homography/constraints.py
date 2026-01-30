#!/usr/bin/env python3
# homography/constraints.py
#
# Find and highlight boundary pixels that are close to each other (within tolerance)
# PLUS: extract and save anchor points along close boundaries, and visualize them.

import os
import json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HIERARCHY_PATH = os.path.join(ROOT, "sketch", "AEP", "hierarchy_tree.json")
SEG_DIR = os.path.join(ROOT, "sketch", "segmentation_original_image")
VIEWS_DIR = os.path.join(ROOT, "sketch", "views")
OUT_ROOT = os.path.join(ROOT, "sketch", "back_project_masks")

NUM_VIEWS = 6
TOLERANCE_PIXELS = 10  # <-- as requested

# Anchor sampling controls
ANCHORS_PER_SIDE_PER_PAIR = 200   # anchors sampled on each side of a pair (label1 side + label2 side)
ANCHOR_DRAW_RADIUS = 2
ANCHOR_COLOR_L1 = (255, 0, 0)     # BGR: blue
ANCHOR_COLOR_L2 = (0, 0, 255)     # BGR: red
ANCHOR_COLOR_ALL = (0, 255, 255)  # BGR: yellow (for combined vis)


MIN_ANCHOR_SPACING_PX = 10   # minimum distance between sampled anchors
MAX_ANCHORS_PER_SIDE_PER_PAIR = 50  # safety cap

# ----------------------------
# Load hierarchy
# ----------------------------
def load_hierarchy_tree():
    """Load the hierarchy tree JSON."""
    if not os.path.exists(HIERARCHY_PATH):
        return None

    with open(HIERARCHY_PATH, "r") as f:
        return json.load(f)


def extract_parent_child_relationships(hierarchy):
    """Extract all parent-child relationships from the hierarchy tree."""
    relationships = []
    root_label = None

    def traverse(node, parent_label=None):
        nonlocal root_label

        label = node.get("label")
        if label is None:
            return

        if parent_label is None:
            root_label = label
        else:
            relationships.append((parent_label, label))

        children = node.get("children", [])
        for child in children:
            traverse(child, label)

    traverse(hierarchy)

    return relationships, root_label


# ----------------------------
# Anchor sampling helpers
# ----------------------------
def _sample_points_with_min_spacing(bin_mask: np.ndarray,
                                    min_spacing_px: int,
                                    max_points: int) -> np.ndarray:
    """
    Sample (x,y) points from nonzero pixels of bin_mask, enforcing a minimum spacing.
    Returns up to max_points points.
    """
    ys, xs = np.nonzero(bin_mask)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    pts = np.stack([xs, ys], axis=1).astype(np.int32)

    # Shuffle so we get a spread that's not biased by scanline order
    order = np.random.permutation(pts.shape[0])
    pts = pts[order]

    chosen = []
    r2 = float(min_spacing_px * min_spacing_px)

    # Greedy far-enough selection
    for p in pts:
        if not chosen:
            chosen.append(p)
        else:
            # check squared distances to all chosen
            d = np.asarray(chosen) - p[None, :]
            if np.all((d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) >= r2):
                chosen.append(p)

        if len(chosen) >= max_points:
            break

    return np.asarray(chosen, dtype=np.int32)


def _merge_unique_points(points_list, max_points=None) -> np.ndarray:
    """
    Combine multiple (K,2) arrays into a unique set.
    If max_points is not None, subsample to that size.
    """
    if not points_list:
        return np.zeros((0, 2), dtype=np.int32)

    pts = np.concatenate(points_list, axis=0)
    if pts.shape[0] == 0:
        return pts.astype(np.int32)

    # unique rows
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
def find_close_boundary_pixels_and_anchors(boundaries_dict, tolerance_pixels=10, anchors_per_side=200):
    """
    Find boundary pixels from different masks that are close to each other,
    AND extract anchor points on each side of each close pair.

    Returns:
        close_pixels: binary mask (H,W) union of all close boundary pixels
        pair_info: list of dicts with:
            {
              label1, label2,
              num_close_pixels,
              anchors_label1_xy: (K1,2) int array [x,y] on label1 boundary,
              anchors_label2_xy: (K2,2) int array [x,y] on label2 boundary
            }
    """
    labels = list(boundaries_dict.keys())
    H, W = next(iter(boundaries_dict.values())).shape

    close_pixels = np.zeros((H, W), dtype=np.uint8)
    pair_info = []

    print(f"\n  Finding close boundaries (tolerance={tolerance_pixels}px):")

    kernel_size = 2 * tolerance_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for i, label1 in enumerate(labels):
        for label2 in labels[i + 1:]:
            boundary1 = boundaries_dict[label1].astype(np.uint8)
            boundary2 = boundaries_dict[label2].astype(np.uint8)

            # directional closeness:
            # pixels on boundary2 close to boundary1
            dilated1 = cv2.dilate(boundary1, kernel, iterations=1)
            close_on_2 = (dilated1 > 0) & (boundary2 > 0)

            # pixels on boundary1 close to boundary2
            dilated2 = cv2.dilate(boundary2, kernel, iterations=1)
            close_on_1 = (dilated2 > 0) & (boundary1 > 0)

            # union for global visualization
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
                    "anchors_label1_xy": anchors1,  # (K,2) [x,y]
                    "anchors_label2_xy": anchors2,  # (K,2) [x,y]
                })
                print(f"    {label1} <-> {label2}: {num_close} close boundary pixels "
                      f"(anchors: {anchors1.shape[0]} on {label1}, {anchors2.shape[0]} on {label2})")

    return close_pixels, pair_info


# ----------------------------
# Main processing
# ----------------------------
def main():
    print("\n" + "=" * 80)
    print("Finding Close Boundary Pixels (+ Anchor Points)")
    print("=" * 80)
    print(f"Tolerance: {TOLERANCE_PIXELS} pixels\n")

    # Load hierarchy tree
    hierarchy = load_hierarchy_tree()
    if hierarchy is None:
        print(f"ERROR: Hierarchy tree not found: {HIERARCHY_PATH}")
        return

    relationships, root_label = extract_parent_child_relationships(hierarchy)
    print(f"Root component: {root_label}")
    print(f"Parent-Child relationships: {len(relationships)}")
    for parent, child in relationships:
        print(f"  {parent} -> {child}")

    # Process each view
    for x in range(NUM_VIEWS):
        view_name = f"view_{x}"
        print(f"\n{'=' * 80}")
        print(f"Processing {view_name}")
        print(f"{'=' * 80}")

        seg_folder = os.path.join(SEG_DIR, view_name)
        img_path = os.path.join(VIEWS_DIR, f"{view_name}.png")

        if not os.path.exists(img_path):
            print(f"[skip] Missing image: {img_path}")
            continue
        if not os.path.exists(seg_folder):
            print(f"[skip] Missing segmentation folder: {seg_folder}")
            continue

        base = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if base is None:
            print(f"[skip] Failed to read image")
            continue

        H_img, W_img = base.shape[:2]

        # Create output directory
        out_dir = os.path.join(OUT_ROOT, view_name, "constraints")
        os.makedirs(out_dir, exist_ok=True)

        anchor_pairs_dir = os.path.join(out_dir, "anchor_pairs")
        os.makedirs(anchor_pairs_dir, exist_ok=True)

        # Load all masks
        print("\n  Loading masks...")
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

        print(f"    Loaded {len(original_masks)} masks: {list(original_masks.keys())}")

        # Extract boundaries
        kernel = np.ones((3, 3), np.uint8)
        boundaries = {}

        print("\n  Extracting boundaries:")
        for label, mask in original_masks.items():
            binary = (mask > 0).astype(np.uint8)
            eroded = cv2.erode(binary, kernel, iterations=1)
            boundary = binary - eroded
            boundaries[label] = boundary

            num_boundary = int(np.sum(boundary > 0))
            print(f"    {label}: {num_boundary} boundary pixels")

        # Find close boundary pixels + anchors
        close_pixels, pair_info = find_close_boundary_pixels_and_anchors(
            boundaries,
            tolerance_pixels=TOLERANCE_PIXELS,
            anchors_per_side=ANCHORS_PER_SIDE_PER_PAIR,
        )

        total_close = int(np.sum(close_pixels > 0))
        print(f"\n  Total close boundary pixels: {total_close}")
        print(f"  Found {len(pair_info)} mask pairs with close boundaries")

        # Visualization 1: masks white, boundaries red, close boundaries green
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

        out_path1 = os.path.join(out_dir, "boundaries_with_close_highlighted.png")
        cv2.imwrite(out_path1, vis1)
        print(f"\n  Saved: boundaries_with_close_highlighted.png")
        print(f"    White = masks")
        print(f"    Red = all boundaries")
        print(f"    GREEN = close boundaries (within {TOLERANCE_PIXELS}px)")

        # Visualization 2: close boundary pixels only
        vis2 = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        vis2[close_pixels > 0] = (0, 255, 0)
        out_path2 = os.path.join(out_dir, "close_boundaries_only.png")
        cv2.imwrite(out_path2, vis2)
        print(f"  Saved: close_boundaries_only.png")

        # Build per-label anchor unions (useful later for global solve per label)
        label_to_anchor_points = {lbl: [] for lbl in original_masks.keys()}

        # Per-pair anchor visualizations + convert anchors for JSON
        close_pairs_json = []
        for p in pair_info:
            l1 = p["label1"]
            l2 = p["label2"]
            anchors1 = p["anchors_label1_xy"]
            anchors2 = p["anchors_label2_xy"]

            label_to_anchor_points[l1].append(anchors1)
            label_to_anchor_points[l2].append(anchors2)

            # Pair overlay image
            pair_vis = np.zeros((H_img, W_img, 3), dtype=np.uint8)

            # show boundaries for the two labels
            pair_vis[boundaries[l1] > 0] = (255, 0, 0)   # blue
            pair_vis[boundaries[l2] > 0] = (0, 0, 255)   # red

            # show close pixels (green) as background highlight
            # (only those close between these two, recompute quickly using same dilation logic)
            kernel_size = 2 * TOLERANCE_PIXELS + 1
            kernel_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dil1 = cv2.dilate(boundaries[l1].astype(np.uint8), kernel_tol, iterations=1)
            dil2 = cv2.dilate(boundaries[l2].astype(np.uint8), kernel_tol, iterations=1)
            close_on_2 = (dil1 > 0) & (boundaries[l2] > 0)
            close_on_1 = (dil2 > 0) & (boundaries[l1] > 0)
            close_union = close_on_1 | close_on_2
            pair_vis[close_union] = (0, 255, 0)

            # draw anchors
            _draw_points(pair_vis, anchors1, ANCHOR_COLOR_L1, radius=ANCHOR_DRAW_RADIUS)
            _draw_points(pair_vis, anchors2, ANCHOR_COLOR_L2, radius=ANCHOR_DRAW_RADIUS)

            pair_path = os.path.join(anchor_pairs_dir, f"{l1}__{l2}.png")
            cv2.imwrite(pair_path, pair_vis)

            close_pairs_json.append({
                "label1": l1,
                "label2": l2,
                "num_close_pixels": int(p["num_close_pixels"]),
                "anchors_label1_xy": anchors1.astype(int).tolist(),  # [[x,y],...]
                "anchors_label2_xy": anchors2.astype(int).tolist(),  # [[x,y],...]
                "pair_anchor_vis": os.path.relpath(pair_path, out_dir),
            })

        # Per-label merged anchors (unique)
        label_anchors_json = {}
        for lbl, pts_list in label_to_anchor_points.items():
            merged = _merge_unique_points(pts_list, max_points=2000)  # cap to avoid huge json
            label_anchors_json[lbl] = merged.astype(int).tolist()

        # Visualization: all anchors over vis1
        all_anchor_vis = vis1.copy()
        # union all points across labels (cap)
        all_pts = _merge_unique_points([np.array(v, dtype=np.int32) for v in label_anchors_json.values() if len(v) > 0],
                                       max_points=5000)
        _draw_points(all_anchor_vis, all_pts, ANCHOR_COLOR_ALL, radius=ANCHOR_DRAW_RADIUS)
        out_path3 = os.path.join(out_dir, "anchor_points_all.png")
        cv2.imwrite(out_path3, all_anchor_vis)
        print(f"  Saved: anchor_points_all.png (all anchors in yellow)")

        # Save JSON summary (now includes anchors)
        summary = {
            "view": view_name,
            "tolerance_pixels": int(TOLERANCE_PIXELS),
            "anchors_per_side_per_pair": int(ANCHORS_PER_SIDE_PER_PAIR),
            "total_close_boundary_pixels": int(total_close),
            "num_pairs_with_close_boundaries": int(len(close_pairs_json)),
            "close_pairs": close_pairs_json,
            "label_anchor_points_xy": label_anchors_json,  # per-label union anchors [[x,y],...]
            "outputs": {
                "boundaries_with_close_highlighted": "boundaries_with_close_highlighted.png",
                "close_boundaries_only": "close_boundaries_only.png",
                "anchor_points_all": "anchor_points_all.png",
                "anchor_pairs_dir": "anchor_pairs/",
            }
        }

        summary_path = os.path.join(out_dir, "close_boundaries_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: close_boundaries_summary.json (now includes anchor points)")

        print(f"\n  All outputs saved to: {out_dir}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


# This module is called from 12_z_2d_homography.py
if __name__ == "__main__":
    main()
