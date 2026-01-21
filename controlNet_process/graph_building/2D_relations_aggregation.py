#!/usr/bin/env python3
"""
graph_building/2D_relations_aggregation.py

Compute aggregated 2D neighbor relations across all views.

Mask format in each view folder:
  {label}_{i}_mask.png
Mask semantics:
  white/non-zero = selected, black = not selected

Neighbor rule (10% tolerance):
  For masks A,B:
    tol_px = ceil(0.10 * min(diag(A_bbox), diag(B_bbox)))  (>=1)
  A and B are neighbors if dilate(A, tol_px) intersects B (or vice versa).

This module does NOT print and does NOT save.
"""

import os
import re
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image


FNAME_RE = re.compile(r"^(?P<label>.+)_(?P<idx>\d+)_mask\.png$", re.IGNORECASE)


def _load_mask(path: str) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img > 0
    img = Image.open(path).convert("L")
    return (np.array(img) > 0)


def _bbox_from_mask(m: np.ndarray):
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (y0, y1, x0, x1)


def _bbox_diag(b) -> float:
    y0, y1, x0, x1 = b
    return float(np.sqrt((y1 - y0 + 1) ** 2 + (x1 - x0 + 1) ** 2))


def _bboxes_maybe_close(b1, b2, pad: int) -> bool:
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


def _dilate_mask(m: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return m

    if cv2 is None:
        # naive square dilation fallback
        k = 2 * radius_px + 1
        padded = np.pad(
            m.astype(np.uint8),
            ((radius_px, radius_px), (radius_px, radius_px)),
            mode="constant",
        )
        out = np.zeros_like(m, dtype=np.uint8)
        for dy in range(k):
            for dx in range(k):
                out = np.maximum(out, padded[dy:dy + m.shape[0], dx:dx + m.shape[1]])
        return out.astype(bool)

    k = 2 * radius_px + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    d = cv2.dilate(m.astype(np.uint8), kernel, iterations=1)
    return d.astype(bool)


def compute_all_view_relations(
    seg_root: str,
    views=range(6),
    tol_ratio: float = 0.10,
):
    """
    Args:
      seg_root: path to sketch/segmentation_original_image
      views: iterable of view indices (default 0..5)
      tol_ratio: tolerance ratio (default 0.10)

    Returns:
      relations_all: set of undirected pairs (a,b) with a<b where a,b are mask_ids "{label}_{idx}"
      stats: dict with basic counts
    """
    relations_all = set()
    views_found = 0
    total_masks = 0

    for x in views:
        vd = os.path.join(seg_root, f"view_{x}")
        if not os.path.isdir(vd):
            continue
        views_found += 1

        files = [f for f in os.listdir(vd) if f.lower().endswith("_mask.png")]
        items = []
        for f in sorted(files):
            m = FNAME_RE.match(f)
            if not m:
                continue
            label = m.group("label")
            idx = int(m.group("idx"))
            mask_id = f"{label}_{idx}"
            items.append((mask_id, os.path.join(vd, f)))

        masks = {}
        bboxes = {}
        for mid, path in items:
            bm = _load_mask(path)
            masks[mid] = bm
            bboxes[mid] = _bbox_from_mask(bm)

        mask_ids = sorted(masks.keys())
        total_masks += len(mask_ids)

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

                tol_px = int(np.ceil(tol_ratio * min(_bbox_diag(bb_a), _bbox_diag(bb_b))))
                tol_px = max(tol_px, 1)

                if not _bboxes_maybe_close(bb_a, bb_b, pad=tol_px):
                    continue

                dil_a = _dilate_mask(masks[a], tol_px)
                dil_b = _dilate_mask(masks[b], tol_px)

                if np.any(dil_a & masks[b]) or np.any(dil_b & masks[a]):
                    relations_all.add(tuple(sorted((a, b))))

    stats = {
        "views_found": views_found,
        "total_masks_seen": total_masks,
        "relations_all_count": len(relations_all),
        "tol_ratio": tol_ratio,
    }
    return relations_all, stats
