#!/usr/bin/env python3
"""
1) Compute 2D neighbor pairs from masks in:
   sketch/segmentation_original_image/view_{0..5}/{label}_{i}_mask.png

   Neighbor rule (10% tolerance):
     For masks A,B:
       tol_px = ceil(0.10 * min(diag(A_bbox), diag(B_bbox)))  (>=1)
     A and B are neighbors if dilate(A, tol_px) intersects B (or vice versa).

2) Read sketch/AEP/initial_constraints.json
   Extract all attachment pairs (a,b) from ["attachments"].

3) Print:
   - All attachment pairs (undirected)
   - All 2D neighbor pairs aggregated across views (undirected)
   - Attachment pairs that are NOT in 2D neighbors, excluding any pair where
     either side is "unknown_{x}".

No saving; only print.
"""

import os
import re
import json
import sys
from collections import defaultdict
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image


# ----------------------------
# Config (relative to this script)
# ----------------------------
TOL_RATIO = 0.10
VIEWS = list(range(6))

FNAME_RE = re.compile(r"^(?P<label>.+)_(?P<idx>\d+)_mask\.png$", re.IGNORECASE)


# ----------------------------
# Mask helpers
# ----------------------------
def load_mask(path: str) -> np.ndarray:
    """Return boolean mask (H,W) where True means selected/white."""
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img > 0
    img = Image.open(path).convert("L")
    return (np.array(img) > 0)


def bbox_from_mask(m: np.ndarray):
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (y0, y1, x0, x1)


def bbox_diag(b) -> float:
    y0, y1, x0, x1 = b
    return float(np.sqrt((y1 - y0 + 1) ** 2 + (x1 - x0 + 1) ** 2))


def bboxes_maybe_close(b1, b2, pad: int) -> bool:
    """Quick reject: are bboxes within pad pixels of each other?"""
    y0a, y1a, x0a, x1a = b1
    y0b, y1b, x0b, x1b = b2
    if y0a > y1b + pad:
        return False
    if y0b > y1a + pad:
        return False
    if x0a > x1b + pad:
        return False
    if x0b > x1a + pad:
        return False
    return True


def dilate_mask(m: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return m
    if cv2 is None:
        # naive dilation fallback (square kernel)
        k = 2 * radius_px + 1
        padded = np.pad(
            m.astype(np.uint8),
            ((radius_px, radius_px), (radius_px, radius_px)),
            mode="constant",
        )
        out = np.zeros_like(m, dtype=np.uint8)
        for dy in range(k):
            for dx in range(k):
                out = np.maximum(out, padded[dy : dy + m.shape[0], dx : dx + m.shape[1]])
        return out.astype(bool)

    k = 2 * radius_px + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    d = cv2.dilate(m.astype(np.uint8), kernel, iterations=1)
    return d.astype(bool)


# ----------------------------
# Neighbor computation
# ----------------------------
def compute_neighbor_pairs_for_view(view_dir: str, tol_ratio: float):
    """
    Returns:
      pair_set: set of undirected pairs (a,b) where a < b
      mask_ids: sorted list of mask ids in this view
    """
    files = [f for f in os.listdir(view_dir) if f.lower().endswith("_mask.png")]

    items = []
    for f in sorted(files):
        m = FNAME_RE.match(f)
        if not m:
            continue
        label = m.group("label")
        idx = int(m.group("idx"))
        mask_id = f"{label}_{idx}"
        items.append((mask_id, os.path.join(view_dir, f)))

    masks = {}
    bboxes = {}
    for mid, path in items:
        bm = load_mask(path)
        masks[mid] = bm
        bboxes[mid] = bbox_from_mask(bm)

    mask_ids = sorted(masks.keys())
    pair_set = set()

    for i in range(len(mask_ids)):
        a = mask_ids[i]
        bb_a = bboxes[a]
        if bb_a is None:
            continue
        for j in range(i + 1, len(mask_ids)):
            b = mask_ids[j]
            bb_b = bboxes[b]
            if bb_b is None:
                continue

            tol_px = int(np.ceil(tol_ratio * min(bbox_diag(bb_a), bbox_diag(bb_b))))
            tol_px = max(tol_px, 1)

            if not bboxes_maybe_close(bb_a, bb_b, pad=tol_px):
                continue

            dil_a = dilate_mask(masks[a], tol_px)
            dil_b = dilate_mask(masks[b], tol_px)

            if np.any(dil_a & masks[b]) or np.any(dil_b & masks[a]):
                pair_set.add(tuple(sorted((a, b))))

    return pair_set, mask_ids


def is_unknown(name: str) -> bool:
    return name.startswith("unknown_")


# ----------------------------
# Read attachments from JSON
# ----------------------------
def read_attachment_pairs(constraints_path: str):
    with open(constraints_path, "r") as f:
        data = json.load(f)

    attachments = data.get("attachments", [])
    pair_set = set()
    typed = defaultdict(int)  # relation_type stats (optional print)

    for rel in attachments:
        a = rel.get("a", None)
        b = rel.get("b", None)
        if not a or not b:
            continue
        pair_set.add(tuple(sorted((a, b))))
        rtype = rel.get("relation_type", "unknown_type")
        typed[rtype] += 1

    return pair_set, typed, attachments


# ----------------------------
# Main
# ----------------------------
def main():
    root = os.path.dirname(os.path.abspath(__file__))

    seg_root = os.path.join(root, "sketch", "segmentation_original_image")
    constraints_path = os.path.join(root, "sketch", "AEP", "initial_constraints.json")

    if not os.path.isdir(seg_root):
        print(f"[ERROR] Not found segmentation folder: {seg_root}")
        sys.exit(1)
    if not os.path.isfile(constraints_path):
        print(f"[ERROR] Not found constraints json: {constraints_path}")
        sys.exit(1)

    # 1) Aggregate 2D neighbor pairs across views
    neighbor_pairs_all = set()
    neighbor_pairs_by_view = {}
    masks_by_view = {}

    for x in VIEWS:
        vd = os.path.join(seg_root, f"view_{x}")
        if not os.path.isdir(vd):
            continue
        pairs, mask_ids = compute_neighbor_pairs_for_view(vd, tol_ratio=TOL_RATIO)
        neighbor_pairs_by_view[x] = pairs
        masks_by_view[x] = mask_ids
        neighbor_pairs_all |= pairs

    # 2) Read attachment pairs
    attach_pairs, attach_type_stats, attachments_raw = read_attachment_pairs(constraints_path)

    # 3) Missing attachments: not in 2D neighbors and no unknowns involved
    missing = []
    for a, b in sorted(attach_pairs):
        if is_unknown(a) or is_unknown(b):
            continue
        if (a, b) not in neighbor_pairs_all:
            missing.append((a, b))

    # ----------------------------
    # Print everything
    # ----------------------------
    print("=" * 100)
    print("2D NEIGHBORS FROM MASKS (AGGREGATED)")
    print(f"seg_root: {seg_root}")
    print(f"neighbor_rule: tol_px = ceil({TOL_RATIO} * min(bbox_diag(A), bbox_diag(B)))")
    print("-" * 100)
    print(f"total_views_found: {len(neighbor_pairs_by_view)} (expected up to {len(VIEWS)})")
    for x in sorted(neighbor_pairs_by_view.keys()):
        print(f"  view_{x}: masks={len(masks_by_view[x])}, neighbor_pairs={len(neighbor_pairs_by_view[x])}")
    print(f"TOTAL aggregated neighbor pairs: {len(neighbor_pairs_all)}")
    print("-" * 100)
    for a, b in sorted(neighbor_pairs_all):
        print(f"{a}  <->  {b}")

    print("=" * 100)
    print("ATTACHMENT PAIRS FROM initial_constraints.json")
    print(f"constraints_path: {constraints_path}")
    print("-" * 100)
    print(f"attachment_relations (raw entries): {len(attachments_raw)}")
    print(f"unique attachment pairs (undirected): {len(attach_pairs)}")
    if attach_type_stats:
        print("relation_type stats:")
        for k, v in sorted(attach_type_stats.items(), key=lambda t: (-t[1], t[0])):
            print(f"  {k}: {v}")
    print("-" * 100)
    for a, b in sorted(attach_pairs):
        print(f"{a}  <->  {b}")

    print("=" * 100)
    print("ATTACHMENT PAIRS MISSING IN 2D NEIGHBORS (excluding any unknown_*)")
    print("-" * 100)
    print(f"missing_count: {len(missing)}")
    if not missing:
        print("(none)")
    else:
        for a, b in missing:
            print(f"{a}  <->  {b}")

    print("=" * 100)


if __name__ == "__main__":
    main()
