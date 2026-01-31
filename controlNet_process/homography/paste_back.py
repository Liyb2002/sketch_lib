#!/usr/bin/env python3
# homography/paste_back.py
#
# Paste-back (translation-only) alignment using constraint anchor correspondences.
#
# Inputs per view:
#   sketch/back_project_masks/view_{x}/constraints/close_boundaries_summary.json
#     - close_pairs[*].paired_anchors_l1_l2_xy / paired_anchors_l2_l1_xy
#   sketch/back_project_masks/view_{x}/homography/{label}_mask_warped.png
#   sketch/views/view_{x}.png
#   sketch/AEP/hierarchy_tree.json  (dict format: label -> {parent, children})
#
# Outputs per view:
#   sketch/back_project_masks/view_{x}/paste_back/
#     - edges/{parent}__{child}/...
#     - all_masks_after_overlay.png
#     - paste_back_results.json

import os
import json
from collections import deque
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HIER_PATH = os.path.join(ROOT, "sketch", "AEP", "hierarchy_tree.json")
VIEWS_DIR = os.path.join(ROOT, "sketch", "views")
SEG_DIR = os.path.join(ROOT, "sketch", "segmentation_original_image")
BPM_DIR = os.path.join(ROOT, "sketch", "back_project_masks")

NUM_VIEWS = 6

# Visual colors (BGR)
COL_PARENT = (255, 0, 0)     # blue
COL_CHILD  = (0, 0, 255)     # red
COL_LINE   = (255, 255, 0)   # cyan/yellow-ish
COL_CLOSE  = (0, 255, 0)     # green

ANCHOR_RADIUS = 2
LINE_THICK = 1
BOUND_THICK = 2
ALPHA = 0.35


# ----------------------------
# IO + hierarchy
# ----------------------------
def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_hierarchy_tree(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError("hierarchy_tree.json must be dict(label -> {parent, children})")
    return data


def _find_root_label(tree: Dict[str, Any]) -> str:
    roots = []
    for lbl, info in tree.items():
        if info.get("parent", None) is None:
            roots.append(lbl)
    if len(roots) == 0:
        # fallback: pick arbitrary label
        return next(iter(tree.keys()))
    # if multiple roots, pick first deterministic
    roots.sort()
    return roots[0]


def _bfs_edges_from_root(tree: Dict[str, Any], root: str) -> List[Tuple[str, str]]:
    """Return directed edges (parent, child) in BFS order from root."""
    edges = []
    q = deque([root])
    seen = set([root])
    while q:
        p = q.popleft()
        children = tree.get(p, {}).get("children", []) or []
        for c in children:
            edges.append((p, c))
            if c not in seen:
                seen.add(c)
                q.append(c)
    return edges


# ----------------------------
# Mask + geometry utils
# ----------------------------
def _imread_gray(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def _ensure_mask_shape(mask_u8: np.ndarray, H: int, W: int) -> np.ndarray:
    if mask_u8.shape[:2] != (H, W):
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    return mask_u8


def _mask_boundary(mask_u8: np.ndarray) -> np.ndarray:
    bin_u8 = (mask_u8 > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    er = cv2.erode(bin_u8, kernel, iterations=1)
    b = (bin_u8 - er)
    return (b > 0).astype(np.uint8) * 255


def _translate_mask(mask_u8: np.ndarray, dx: float, dy: float) -> np.ndarray:
    H, W = mask_u8.shape[:2]
    M = np.array([[1.0, 0.0, dx],
                  [0.0, 1.0, dy]], dtype=np.float32)
    out = cv2.warpAffine(
        mask_u8,
        M,
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    out = (out > 0).astype(np.uint8) * 255
    return out


def _translate_points_xy(pts_xy: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if pts_xy.size == 0:
        return pts_xy.astype(np.float64)
    pts = pts_xy.astype(np.float64).copy()
    pts[:, 0] += dx
    pts[:, 1] += dy
    return pts


def _draw_points(img_bgr: np.ndarray, pts_xy: np.ndarray, color: Tuple[int, int, int], r: int = 2):
    for x, y in pts_xy:
        cv2.circle(img_bgr, (int(round(x)), int(round(y))), r, color, thickness=-1)


def _draw_pairs(img_bgr: np.ndarray,
                parent_xy: np.ndarray,
                child_xy: np.ndarray,
                color_line: Tuple[int, int, int],
                thickness: int = 1):
    K = min(parent_xy.shape[0], child_xy.shape[0])
    for i in range(K):
        x1, y1 = parent_xy[i]
        x2, y2 = child_xy[i]
        cv2.line(img_bgr,
                 (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))),
                 color_line,
                 thickness)


def _overlay_boundary(img_bgr: np.ndarray, boundary_u8: np.ndarray, color: Tuple[int, int, int], thick: int = 2):
    out = img_bgr.copy()
    contours, _ = cv2.findContours(boundary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color, thick)
    return out


def _overlay_mask_fill(img_bgr: np.ndarray, mask_u8: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.35):
    out = img_bgr.copy()
    bin_ = mask_u8 > 0
    if np.any(bin_):
        fill = np.zeros_like(out)
        fill[:] = color
        out[bin_] = cv2.addWeighted(out[bin_], 1.0 - alpha, fill[bin_], alpha, 0)
    return out


def _mean_pair_dist(parent_xy: np.ndarray, child_xy: np.ndarray) -> float:
    if parent_xy.size == 0 or child_xy.size == 0:
        return float("nan")
    K = min(parent_xy.shape[0], child_xy.shape[0])
    d = parent_xy[:K] - child_xy[:K]
    dist = np.sqrt(np.sum(d * d, axis=1))
    return float(np.mean(dist))


# ----------------------------
# Constraints: get paired anchors for an edge
# ----------------------------
def _build_pair_index(constraints_json: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Map unordered (a,b) -> close_pair dict.
    """
    out = {}
    for p in constraints_json.get("close_pairs", []):
        a = p.get("label1")
        b = p.get("label2")
        if a is None or b is None:
            continue
        key = tuple(sorted([a, b]))
        out[key] = p
    return out


def _get_paired_anchors_for_edge(pair_rec: Dict[str, Any], parent: str, child: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (parent_pts_xy, child_pts_xy) with guaranteed correspondence ordering.
    Requires that constraints JSON saved paired_anchors_l1_l2_xy and paired_anchors_l2_l1_xy.
    """
    l1 = pair_rec.get("label1")
    l2 = pair_rec.get("label2")

    p12 = np.array(pair_rec.get("paired_anchors_l1_l2_xy", []), dtype=np.float64)  # [[[x1,y1],[x2,y2]]]
    p21 = np.array(pair_rec.get("paired_anchors_l2_l1_xy", []), dtype=np.float64)

    # Choose the correct direction so returned arrays are (parent, child)
    if parent == l1 and child == l2:
        pairs = p12
        # pairs[:,0] are l1 points, pairs[:,1] are l2 points
        parent_xy = pairs[:, 0, :] if pairs.size else np.zeros((0, 2), dtype=np.float64)
        child_xy  = pairs[:, 1, :] if pairs.size else np.zeros((0, 2), dtype=np.float64)
        return parent_xy, child_xy

    if parent == l2 and child == l1:
        pairs = p21
        # pairs[:,0] are l2 points, pairs[:,1] are l1 points
        parent_xy = pairs[:, 0, :] if pairs.size else np.zeros((0, 2), dtype=np.float64)
        child_xy  = pairs[:, 1, :] if pairs.size else np.zeros((0, 2), dtype=np.float64)
        return parent_xy, child_xy

    # If something inconsistent, return empty
    return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)


# ----------------------------
# Main paste-back
# ----------------------------
def main():
    tree = _load_hierarchy_tree(HIER_PATH)
    root = _find_root_label(tree)
    edges = _bfs_edges_from_root(tree, root)

    for vx in range(NUM_VIEWS):
        view = f"view_{vx}"
        base_path = os.path.join(VIEWS_DIR, f"{view}.png")
        if not os.path.exists(base_path):
            continue

        base = cv2.imread(base_path, cv2.IMREAD_COLOR)
        if base is None:
            continue
        H_img, W_img = base.shape[:2]

        constraints_path = os.path.join(BPM_DIR, view, "constraints", "close_boundaries_summary.json")
        if not os.path.exists(constraints_path):
            continue
        constraints_json = _read_json(constraints_path)
        pair_index = _build_pair_index(constraints_json)

        # output dirs
        out_dir = os.path.join(BPM_DIR, view, "paste_back")
        edges_dir = os.path.join(out_dir, "edges")
        os.makedirs(edges_dir, exist_ok=True)

        # Load initial masks: prefer homography warped, else original segmentation
        masks: Dict[str, np.ndarray] = {}
        hom_dir = os.path.join(BPM_DIR, view, "homography")
        seg_dir = os.path.join(SEG_DIR, view)

        # candidate labels from constraints json (safer than listing folders)
        labels = set()
        for p in constraints_json.get("close_pairs", []):
            if "label1" in p: labels.add(p["label1"])
            if "label2" in p: labels.add(p["label2"])
        labels.add(root)

        for lbl in sorted(labels):
            warped_path = os.path.join(hom_dir, f"{lbl}_mask_warped.png")
            if os.path.exists(warped_path):
                m = _imread_gray(warped_path)
            else:
                m = _imread_gray(os.path.join(seg_dir, f"{lbl}_mask.png"))
            if m is None:
                continue
            masks[lbl] = _ensure_mask_shape(m, H_img, W_img)

        if root not in masks:
            # can't do anything meaningful
            continue

        # Track moved masks separately (so child-of-child uses already-moved parent)
        moved_masks: Dict[str, np.ndarray] = dict(masks)

        results = {
            "view": view,
            "root": root,
            "edges_processed": [],
        }

        # Process BFS edges: parent fixed, child moved
        for parent, child in edges:
            if parent not in moved_masks or child not in moved_masks:
                continue

            key = tuple(sorted([parent, child]))
            if key not in pair_index:
                # no constraint record between these two => skip for now
                continue

            pair_rec = pair_index[key]
            parent_pts, child_pts = _get_paired_anchors_for_edge(pair_rec, parent, child)

            if parent_pts.shape[0] == 0 or child_pts.shape[0] == 0:
                continue

            # Compute translation (child -> parent) from paired anchors
            # dx,dy = mean(parent - child)
            delta = parent_pts - child_pts
            dx = float(np.mean(delta[:, 0]))
            dy = float(np.mean(delta[:, 1]))

            child_before = moved_masks[child].copy()
            parent_mask  = moved_masks[parent]  # unchanged

            child_after = _translate_mask(child_before, dx, dy)
            moved_masks[child] = child_after

            # -------- visuals: correspondence BEFORE / AFTER (using given pairs) --------
            edge_out = os.path.join(edges_dir, f"{parent}__{child}")
            os.makedirs(edge_out, exist_ok=True)

            # BEFORE: show boundaries + anchors + correspondence lines
            vis_before = base.copy()
            vis_before = _overlay_boundary(vis_before, _mask_boundary(parent_mask), COL_PARENT, thick=BOUND_THICK)
            vis_before = _overlay_boundary(vis_before, _mask_boundary(child_before),  COL_CHILD,  thick=BOUND_THICK)
            _draw_points(vis_before, parent_pts, COL_PARENT, r=ANCHOR_RADIUS)
            _draw_points(vis_before, child_pts,  COL_CHILD,  r=ANCHOR_RADIUS)
            _draw_pairs(vis_before, parent_pts, child_pts, COL_LINE, thickness=LINE_THICK)
            cv2.imwrite(os.path.join(edge_out, "correspondence_before.png"), vis_before)

            # AFTER: child anchors translated
            child_pts_after = _translate_points_xy(child_pts, dx, dy)
            vis_after = base.copy()
            vis_after = _overlay_boundary(vis_after, _mask_boundary(parent_mask), COL_PARENT, thick=BOUND_THICK)
            vis_after = _overlay_boundary(vis_after, _mask_boundary(child_after),  COL_CHILD,  thick=BOUND_THICK)
            _draw_points(vis_after, parent_pts, COL_PARENT, r=ANCHOR_RADIUS)
            _draw_points(vis_after, child_pts_after, COL_CHILD, r=ANCHOR_RADIUS)
            _draw_pairs(vis_after, parent_pts, child_pts_after, COL_LINE, thickness=LINE_THICK)
            cv2.imwrite(os.path.join(edge_out, "correspondence_after.png"), vis_after)

            # Optional: filled overlays before/after (more legible)
            filled_before = base.copy()
            filled_before = _overlay_mask_fill(filled_before, parent_mask, COL_PARENT, alpha=ALPHA)
            filled_before = _overlay_mask_fill(filled_before, child_before, COL_CHILD, alpha=ALPHA)
            cv2.imwrite(os.path.join(edge_out, "masks_filled_before.png"), filled_before)

            filled_after = base.copy()
            filled_after = _overlay_mask_fill(filled_after, parent_mask, COL_PARENT, alpha=ALPHA)
            filled_after = _overlay_mask_fill(filled_after, child_after, COL_CHILD, alpha=ALPHA)
            cv2.imwrite(os.path.join(edge_out, "masks_filled_after.png"), filled_after)

            # Save child before/after masks
            cv2.imwrite(os.path.join(edge_out, f"{child}_mask_before.png"), child_before)
            cv2.imwrite(os.path.join(edge_out, f"{child}_mask_after.png"), child_after)

            # Errors (mean paired distance)
            err_before = _mean_pair_dist(parent_pts, child_pts)
            err_after  = _mean_pair_dist(parent_pts, child_pts_after)

            results["edges_processed"].append({
                "parent": parent,
                "child": child,
                "translation_dx_dy": [dx, dy],
                "num_pairs_used": int(parent_pts.shape[0]),
                "mean_pair_dist_before": err_before,
                "mean_pair_dist_after": err_after,
                "outputs": {
                    "correspondence_before": os.path.join("edges", f"{parent}__{child}", "correspondence_before.png"),
                    "correspondence_after":  os.path.join("edges", f"{parent}__{child}", "correspondence_after.png"),
                    "masks_filled_before":   os.path.join("edges", f"{parent}__{child}", "masks_filled_before.png"),
                    "masks_filled_after":    os.path.join("edges", f"{parent}__{child}", "masks_filled_after.png"),
                },
            })

        # Final overlay of all moved masks
        all_overlay = base.copy()
        for lbl, m in moved_masks.items():
            # deterministic pseudo-color by hash
            h = (abs(hash(lbl)) % 180)
            hsv = np.uint8([[[h, 220, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            col = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            all_overlay = _overlay_mask_fill(all_overlay, m, col, alpha=0.25)
        cv2.imwrite(os.path.join(out_dir, "all_masks_after_overlay.png"), all_overlay)

        # Save json
        results["outputs"] = {
            "all_masks_after_overlay": "all_masks_after_overlay.png",
        }
        with open(os.path.join(out_dir, "paste_back_results.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
