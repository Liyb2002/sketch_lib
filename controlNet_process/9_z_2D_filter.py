#!/usr/bin/env python3
"""
Compute 2D neighbors among masks in:
  sketch/segmentation_original_image/view_{0..5}/

Mask format:
  {label}_{i}_mask.png

Neighbor definition (10% tolerance):
  Two masks A and B are neighbors if, after dilating each mask by a tolerance
  radius tol_px = ceil(0.10 * min(diag(A_bbox), diag(B_bbox))), they intersect.
  (We also prune by bbox distance with the same tol_px.)

No saving: only prints.
"""

import os
import re
import sys
from collections import defaultdict
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image


FNAME_RE = re.compile(r"^(?P<label>.+)_(?P<idx>\d+)_mask\.png$", re.IGNORECASE)


def load_mask(path: str) -> np.ndarray:
    """Return boolean mask (H,W) where True means selected/white."""
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img > 0  # robust to anti-aliasing
    else:
        img = Image.open(path).convert("L")
        arr = np.array(img)
        return arr > 0


def bbox_from_mask(m: np.ndarray):
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (y0, y1, x0, x1)


def bbox_diag(b) -> float:
    """Diagonal length of bbox in pixels."""
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
    """Binary dilation with square kernel radius_px (>=0)."""
    if radius_px <= 0:
        return m

    if cv2 is None:
        # PIL-only fallback: naive square dilation via max over shifted windows
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


def compute_neighbors_for_view(view_dir: str, tol_ratio: float = 0.10):
    """
    Returns:
      neighbors: dict mask_id -> sorted list of neighbor mask_ids
      mask_ids: list of mask_ids in this view
    mask_id is "{label}_{idx}" (parsed from filename).
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
    for mask_id, path in items:
        bm = load_mask(path)
        masks[mask_id] = bm
        bboxes[mask_id] = bbox_from_mask(bm)

    neighbors = {mid: set() for mid in masks.keys()}
    mask_ids = list(masks.keys())

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

            # 10% tolerance based on bbox diagonal
            tol_px = int(np.ceil(tol_ratio * min(bbox_diag(bb_a), bbox_diag(bb_b))))
            tol_px = max(tol_px, 1)

            # bbox prune: if far beyond tol, skip
            if not bboxes_maybe_close(bb_a, bb_b, pad=tol_px):
                continue

            # dilate & test (symmetric, but we do both to be safe)
            dil_a = dilate_mask(masks[a], tol_px)
            dil_b = dilate_mask(masks[b], tol_px)

            if np.any(dil_a & masks[b]) or np.any(dil_b & masks[a]):
                neighbors[a].add(b)
                neighbors[b].add(a)

    neighbors_sorted = {k: sorted(v) for k, v in neighbors.items()}
    return neighbors_sorted, sorted(mask_ids)


def main():
    root = os.path.dirname(os.path.abspath(__file__))

    seg_root = os.path.join(root, "sketch", "segmentation_original_image")
    if not os.path.isdir(seg_root):
        print(f"[ERROR] Not found: {seg_root}")
        sys.exit(1)

    # Collect views view_0..view_5 if present
    all_views = []
    for x in range(6):
        vd = os.path.join(seg_root, f"view_{x}")
        if os.path.isdir(vd):
            all_views.append((x, vd))

    if not all_views:
        print(f"[ERROR] No view_*/ folders found under: {seg_root}")
        sys.exit(1)

    tol_ratio = 0.10

    # Aggregation structures (across views)
    pair_view_count = defaultdict(int)   # (a,b) -> number of views where they are neighbors
    per_mask_global = defaultdict(set)   # mask_id -> set of neighbor mask_ids (union across views)

    for x, vd in all_views:
        neighbors, mask_ids = compute_neighbors_for_view(vd, tol_ratio=tol_ratio)

        print("=" * 80)
        print(f"VIEW view_{x}  ({vd})")
        print(f"masks: {len(mask_ids)}   neighbor_rule: tol={int(tol_ratio*100)}% of min bbox diagonal")
        print("-" * 80)

        # print per-mask neighbors
        for mid in mask_ids:
            nbrs = neighbors.get(mid, [])
            print(f"{mid}: {', '.join(nbrs) if nbrs else '(no neighbors)'}")

        # aggregate pairs for this view (count each pair once per view)
        seen_pairs = set()
        for a in mask_ids:
            for b in neighbors.get(a, []):
                key = tuple(sorted((a, b)))
                seen_pairs.add(key)
        for key in seen_pairs:
            pair_view_count[key] += 1

        # union per-mask neighbors across views
        for a, nbrs in neighbors.items():
            for b in nbrs:
                per_mask_global[a].add(b)

    print("=" * 80)
    print("AGGREGATED ACROSS ALL VIEWS")
    print("-" * 80)

    # 1) Per-mask union neighbors
    print("Per-mask neighbor union (across views):")
    if not per_mask_global:
        print("(no masks found)")
    else:
        for a in sorted(per_mask_global.keys()):
            nbrs = sorted(per_mask_global[a])
            print(f"{a}: {', '.join(nbrs) if nbrs else '(no neighbors)'}")

    print("-" * 80)
    # 2) Pair counts (how many views this pair appears as neighbors)
    print("Neighbor pair counts (#views where the pair are neighbors):")
    if not pair_view_count:
        print("(no neighboring pairs found)")
    else:
        for (a, b), cnt in sorted(pair_view_count.items(), key=lambda t: (-t[1], t[0][0], t[0][1])):
            print(f"{a}  <->  {b} : {cnt}")

    print("=" * 80)


if __name__ == "__main__":
    main()
