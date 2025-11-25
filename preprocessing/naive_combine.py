#!/usr/bin/env python
"""
naive_combine.py

Replace a shared component label (e.g. 'wheel') in one sketch
with the corresponding components taken from another sketch.

Handles multiple disconnected parts (e.g., two wheels):
  - Finds connected components in source and destination masks.
  - Matches them one-to-one by nearest centroid.
  - For each matched pair:
      * remove original dst component (white-out its region)
      * resize src component (non-uniform) to match dst bbox exactly
      * paste src component into dst at that bbox

Folder structure:
  sketches/
    0/
      plain.png
      wheel.png
      ...
    1/
      plain.png
      wheel.png
      ...

Config:
  SRC_ID, DST_ID, LABEL below.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image

# ----------------- config -----------------
ROOT_DIR = Path("sketches")

# Source & destination sketch folders
SRC_ID = "0"
DST_ID = "1"

# Label name shared by both sketches
LABEL = "tank"

# Diff threshold for mask extraction
DIFF_THRESH = 20

# Padding for cropping source components (destination bboxes use NO padding)
SRC_PAD = 2


# ----------------- helpers -----------------
def compute_mask(plain_img: Image.Image,
                 labeled_img: Image.Image,
                 diff_thresh: int = DIFF_THRESH) -> np.ndarray:
    """
    Compute a binary mask (H, W) where labeled_img differs from plain_img.
    Returns uint8 array with values 0 or 1.
    """
    if plain_img.size != labeled_img.size:
        raise ValueError(f"Image sizes do not match: {plain_img.size} vs {labeled_img.size}")

    arr_plain = np.asarray(plain_img.convert("RGB"), dtype=np.int16)
    arr_label = np.asarray(labeled_img.convert("RGB"), dtype=np.int16)

    diff = np.abs(arr_label - arr_plain)
    diff_mag = diff.sum(axis=2)
    mask = (diff_mag > diff_thresh).astype(np.uint8)
    return mask


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """
    Simple 4-connected component labeling on a binary mask.
    Returns a list of masks (same shape), one per component.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 1 and not visited[y, x]:
                # BFS/DFS to collect this component
                stack = [(y, x)]
                visited[y, x] = True
                coords = []

                while stack:
                    cy, cx = stack.pop()
                    coords.append((cy, cx))

                    # 4-neighborhood
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                                   (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] == 1 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))

                comp_mask = np.zeros_like(mask, dtype=np.uint8)
                ys, xs = zip(*coords)
                comp_mask[ys, xs] = 1
                components.append(comp_mask)

    return components


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute tight bounding box (x_min, y_min, x_max, y_max) where mask == 1.
    Returns None if the mask is empty.
    """
    ys, xs = np.where(mask == 1)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return int(x_min), int(y_min), int(x_max), int(y_max)


def centroid_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    """
    Compute (cx, cy) centroid in image coordinates for mask == 1.
    """
    ys, xs = np.where(mask == 1)
    cx = xs.mean()
    cy = ys.mean()
    return float(cx), float(cy)


def component_from_mask(
    plain_img: Image.Image,
    mask: np.ndarray,
    pad: int = SRC_PAD
) -> Optional[Image.Image]:
    """
    From plain sketch + mask, return an RGBA component image cropped around the mask.
    If mask is empty, returns None.
    """
    ys, xs = np.where(mask == 1)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w, h = plain_img.size
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w - 1, x_max + pad)
    y_max = min(h - 1, y_max + pad)

    plain_rgba = plain_img.convert("RGBA")
    arr_plain = np.asarray(plain_rgba)

    comp_arr = np.zeros_like(arr_plain)
    comp_arr[..., 3] = 0  # fully transparent

    comp_arr[mask == 1] = arr_plain[mask == 1]
    comp_arr_cropped = comp_arr[y_min:y_max + 1, x_min:x_max + 1]

    comp_img = Image.fromarray(comp_arr_cropped, mode="RGBA")
    return comp_img


def load_plain_and_label(folder: Path, label: str) -> Tuple[Image.Image, Image.Image]:
    plain_path = folder / "plain.png"
    label_path = folder / f"{label}.png"

    if not plain_path.is_file():
        raise FileNotFoundError(f"{plain_path} not found")
    if not label_path.is_file():
        raise FileNotFoundError(f"{label_path} not found")

    return Image.open(plain_path), Image.open(label_path)


def match_components_by_centroid(
    src_centroids: List[Tuple[float, float]],
    dst_centroids: List[Tuple[float, float]],
) -> List[Tuple[int, int]]:
    """
    Greedy nearest-neighbor matching:
      - for each dst centroid, pick closest unused src centroid.
    Returns list of pairs (src_idx, dst_idx).
    """
    matches: List[Tuple[int, int]] = []
    src_used = [False] * len(src_centroids)

    for dst_idx, (dcx, dcy) in enumerate(dst_centroids):
        best_src = None
        best_d2 = float("inf")
        for src_idx, (scx, scy) in enumerate(src_centroids):
            if src_used[src_idx]:
                continue
            dx = scx - dcx
            dy = scy - dcy
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_src = src_idx
        if best_src is not None:
            src_used[best_src] = True
            matches.append((best_src, dst_idx))

    return matches


# ----------------- main pipeline -----------------
def naive_combine(
    root: Path,
    src_id: str,
    dst_id: str,
    label: str,
) -> Path:
    """Replace `label` components in dst sketch with those from src sketch."""
    src_folder = root / src_id
    dst_folder = root / dst_id

    # --- load images ---
    src_plain, src_label_img = load_plain_and_label(src_folder, label)
    dst_plain, dst_label_img = load_plain_and_label(dst_folder, label)

    # --- compute masks ---
    src_mask = compute_mask(src_plain, src_label_img)
    dst_mask = compute_mask(dst_plain, dst_label_img)

    # --- connected components (multiple wheels, etc.) ---
    src_comps_masks = connected_components(src_mask)
    dst_comps_masks = connected_components(dst_mask)

    if len(src_comps_masks) == 0:
        raise RuntimeError(f"No source components found for label '{label}' in {src_folder}")
    if len(dst_comps_masks) == 0:
        raise RuntimeError(f"No destination components found for label '{label}' in {dst_folder}")

    # --- build src component images and centroids ---
    src_comp_imgs: List[Image.Image] = []
    src_centroids: List[Tuple[float, float]] = []

    for m in src_comps_masks:
        img = component_from_mask(src_plain, m, pad=SRC_PAD)
        if img is None:
            continue
        src_comp_imgs.append(img)
        src_centroids.append(centroid_from_mask(m))

    # --- dst centroids and bboxes ---
    dst_centroids: List[Tuple[float, float]] = []
    dst_bboxes: List[Tuple[int, int, int, int]] = []
    dst_masks_clean: List[np.ndarray] = []

    for m in dst_comps_masks:
        bbox = bbox_from_mask(m)
        if bbox is None:
            continue
        dst_bboxes.append(bbox)
        dst_centroids.append(centroid_from_mask(m))
        dst_masks_clean.append(m)

    if len(src_comp_imgs) == 0 or len(dst_bboxes) == 0:
        raise RuntimeError("Empty components after filtering.")

    # --- match components one-to-one by centroid distance ---
    # Allow up to min(#src, #dst) matches
    matches = match_components_by_centroid(src_centroids, dst_centroids)
    if len(matches) == 0:
        raise RuntimeError("No matches found between source and destination components.")

    # --- prepare destination image array ---
    dst_out = dst_plain.convert("RGBA")
    dst_arr = np.array(dst_out)  # writable array
    white = np.array([255, 255, 255, 255], dtype=np.uint8)

    # 1) remove each matched dest component (white out its own mask)
    for _, dst_idx in matches:
        m = dst_masks_clean[dst_idx]
        mask_bool = m.astype(bool)
        dst_arr[mask_bool] = white

    # convert back to PIL after clearing
    dst_out = Image.fromarray(dst_arr, mode="RGBA")

    # 2) paste each matched source component into corresponding bbox
    for src_idx, dst_idx in matches:
        src_img = src_comp_imgs[src_idx]
        x_min, y_min, x_max, y_max = dst_bboxes[dst_idx]
        dst_w = x_max - x_min + 1
        dst_h = y_max - y_min + 1

        if dst_w <= 0 or dst_h <= 0:
            continue

        # NON-UNIFORM resize: force exact bbox size
        src_resized = src_img.resize((dst_w, dst_h), resample=Image.BICUBIC)

        # Paste so that new component occupies exactly [x_min:x_max, y_min:y_max]
        dst_out.paste(src_resized, (x_min, y_min), src_resized)

    # --- save result ---
    out_dir = root / "combined"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"combined_{src_id}_to_{dst_id}_{label}.png"
    dst_out.save(out_path)
    print(f"[ok] wrote {out_path}")
    return out_path


def main() -> None:
    if not ROOT_DIR.is_dir():
        raise SystemExit(f"Root directory {ROOT_DIR} not found")

    naive_combine(ROOT_DIR, SRC_ID, DST_ID, LABEL)


if __name__ == "__main__":
    main()
