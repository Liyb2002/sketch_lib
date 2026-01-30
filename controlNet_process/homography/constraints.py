#!/usr/bin/env python3
# homography/constraints.py
#
# Find and highlight boundary pixels that are close to each other (within tolerance)

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
TOLERANCE_PIXELS = 5


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
# Find close boundaries
# ----------------------------
def find_close_boundary_pixels(boundaries_dict, tolerance_pixels=5):
    """
    Find boundary pixels from different masks that are close to each other.
    
    Args:
        boundaries_dict: dict mapping label -> boundary mask (binary)
        tolerance_pixels: max distance to consider "close"
    
    Returns:
        close_pixels: binary mask showing pixels that are close to another mask's boundary
        pair_info: list of (label1, label2, num_close_pixels) for reporting
    """
    labels = list(boundaries_dict.keys())
    H, W = next(iter(boundaries_dict.values())).shape
    
    close_pixels = np.zeros((H, W), dtype=np.uint8)
    pair_info = []
    
    print(f"\n  Finding close boundaries (tolerance={tolerance_pixels}px):")
    
    # For each pair of masks
    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            boundary1 = boundaries_dict[label1]
            boundary2 = boundaries_dict[label2]
            
            # Dilate boundary1 by tolerance distance
            kernel_size = 2 * tolerance_pixels + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated1 = cv2.dilate(boundary1, kernel, iterations=1)
            
            # Find where boundary2 intersects with dilated boundary1
            close_region = np.logical_and(dilated1 > 0, boundary2 > 0)
            num_close = np.sum(close_region)
            
            if num_close > 0:
                close_pixels[close_region] = 255
                pair_info.append((label1, label2, num_close))
                print(f"    {label1} <-> {label2}: {num_close} close boundary pixels")
    
    return close_pixels, pair_info


# ----------------------------
# Main processing
# ----------------------------
def main():
    print("\n" + "=" * 80)
    print("Finding Close Boundary Pixels")
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
            
            # Resize if needed
            if mask.shape[:2] != (H_img, W_img):
                mask = cv2.resize(mask, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
            
            original_masks[label] = mask
        
        print(f"    Loaded {len(original_masks)} masks: {list(original_masks.keys())}")
        
        # Extract boundaries for each mask
        kernel = np.ones((3, 3), np.uint8)
        boundaries = {}
        
        print("\n  Extracting boundaries:")
        for label, mask in original_masks.items():
            binary = mask > 0
            eroded = cv2.erode(binary.astype(np.uint8), kernel, iterations=1)
            boundary = binary.astype(np.uint8) - eroded
            boundaries[label] = boundary
            
            num_boundary = np.sum(boundary > 0)
            print(f"    {label}: {num_boundary} boundary pixels")
        
        # Find close boundary pixels
        close_pixels, pair_info = find_close_boundary_pixels(boundaries, tolerance_pixels=TOLERANCE_PIXELS)
        
        total_close = np.sum(close_pixels > 0)
        print(f"\n  Total close boundary pixels: {total_close}")
        print(f"  Found {len(pair_info)} mask pairs with close boundaries")
        
        # Create visualization 1: All masks white, boundaries red, close boundaries highlighted in bright green
        vis1 = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        
        # All masks in white
        all_masks = np.zeros((H_img, W_img), dtype=np.uint8)
        for mask in original_masks.values():
            all_masks[mask > 0] = 255
        vis1[all_masks > 0] = (255, 255, 255)
        
        # All boundaries in red
        all_boundaries = np.zeros((H_img, W_img), dtype=np.uint8)
        for boundary in boundaries.values():
            all_boundaries[boundary > 0] = 255
        vis1[all_boundaries > 0] = (0, 0, 255)
        
        # Close boundaries in BRIGHT GREEN (overwrite red)
        vis1[close_pixels > 0] = (0, 255, 0)
        
        out_path1 = os.path.join(out_dir, "boundaries_with_close_highlighted.png")
        cv2.imwrite(out_path1, vis1)
        print(f"\n  Saved: boundaries_with_close_highlighted.png")
        print(f"    White = masks")
        print(f"    Red = all boundaries")
        print(f"    GREEN = close boundaries (within {TOLERANCE_PIXELS}px)")
        
        # Create visualization 2: Only show close boundary pixels
        vis2 = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        vis2[close_pixels > 0] = (0, 255, 0)
        
        out_path2 = os.path.join(out_dir, "close_boundaries_only.png")
        cv2.imwrite(out_path2, vis2)
        print(f"  Saved: close_boundaries_only.png")
        print(f"    Shows ONLY the close boundary pixels in green")
        
        # Save pair info JSON
        summary = {
            "view": view_name,
            "tolerance_pixels": TOLERANCE_PIXELS,
            "total_close_boundary_pixels": int(total_close),
            "num_pairs_with_close_boundaries": len(pair_info),
            "close_pairs": [
                {
                    "label1": label1,
                    "label2": label2,
                    "num_close_pixels": int(num_close)
                }
                for label1, label2, num_close in pair_info
            ]
        }
        
        summary_path = os.path.join(out_dir, "close_boundaries_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: close_boundaries_summary.json")
        
        print(f"\n  All outputs saved to: {out_dir}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


# This module is called from 12_z_2d_homography.py