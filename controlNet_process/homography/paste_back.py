#!/usr/bin/env python3
# homography/paste_back.py
#
# STEP 3: Paste-back (tree edges, parent fixed; child translated to fit anchors)
#
# For each view:
#   Inputs:
#     - sketch/AEP/hierarchy_tree.json (adjacency dict keyed by label)
#     - sketch/back_project_masks/view_{x}/constraints/close_boundaries_summary.json
#     - sketch/back_project_masks/view_{x}/homography/homography_results.json
#   Outputs:
#     - sketch/back_project_masks/view_{x}/paste_back/
#         root_label.txt
#         paste_back_results_edges.json
#         all_final_masks_overlay.png
#         masks_final/{label}_mask_final.png
#         edges/{parent}__{child}/
#             child_before.png
#             child_after.png
#             overlay_before.png
#             overlay_after.png
#             anchor_fit_before_after.png
#             anchor_fit_lines_before.png
#             anchor_fit_lines_after.png
#             anchors_on_masks_before.png   <-- NEW
#             anchors_on_masks_after.png    <-- NEW
#
# Translation = mean(parent_pts - child_pts) over paired anchors.

import os
import json
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HIERARCHY_PATH = os.path.join(ROOT, "sketch", "AEP", "hierarchy_tree.json")
OUT_ROOT = os.path.join(ROOT, "sketch", "back_project_masks")

NUM_VIEWS = 6

# Vis colors (BGR)
COL_PARENT_MASK = (255, 0, 0)        # blue
COL_CHILD_BEFORE_MASK = (0, 165, 255)  # orange
COL_CHILD_AFTER_MASK = (0, 0, 255)     # red

COL_ANCH_PARENT = (255, 0, 0)        # blue points
COL_ANCH_CHILD_BEFORE = (0, 165, 255)  # orange points
COL_ANCH_CHILD_AFTER = (0, 0, 255)     # red points

COL_LINE = (0, 255, 0)              # green lines

ALPHA = 0.35
CONTOUR_THICK = 2
ANCHOR_RADIUS = 2


# ----------------------------
# IO helpers
# ----------------------------
def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _imread_mask_u8(path: str, H: int, W: int) -> Optional[np.ndarray]:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return ((m > 0).astype(np.uint8) * 255)


def _overlay_mask(img_bgr: np.ndarray, mask_u8: np.ndarray, color_bgr, alpha=0.35, contour_thick=2) -> np.ndarray:
    out = img_bgr.copy()
    binary = mask_u8 > 0
    if not np.any(binary):
        return out

    fill = np.zeros_like(out)
    fill[:] = color_bgr
    out[binary] = cv2.addWeighted(out[binary], 1.0 - alpha, fill[binary], alpha, 0)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color_bgr, contour_thick)
    return out


def _draw_points(img_bgr: np.ndarray, pts_xy: np.ndarray, color_bgr, radius: int = 2):
    for x, y in pts_xy:
        cv2.circle(img_bgr, (int(x), int(y)), radius, color_bgr, thickness=-1)


def _draw_match_lines(img_bgr: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray, color_bgr, thickness: int = 1):
    K = min(a_xy.shape[0], b_xy.shape[0])
    for i in range(K):
        ax, ay = int(a_xy[i, 0]), int(a_xy[i, 1])
        bx, by = int(b_xy[i, 0]), int(b_xy[i, 1])
        cv2.line(img_bgr, (ax, ay), (bx, by), color_bgr, thickness)


# ----------------------------
# Hierarchy helpers (adjacency dict format)
# ----------------------------
def _find_roots(tree: Dict[str, Any]) -> List[str]:
    roots = []
    for label, info in tree.items():
        if isinstance(info, dict) and info.get("parent", None) is None:
            roots.append(label)
    return roots


def _children_of(tree: Dict[str, Any], label: str) -> List[str]:
    info = tree.get(label, {})
    if not isinstance(info, dict):
        return []
    ch = info.get("children", [])
    if ch is None:
        return []
    if isinstance(ch, list):
        return [str(x) for x in ch]
    return []


# ----------------------------
# Anchors + translation
# ----------------------------
def _collect_pair_anchors(constraints: Dict[str, Any]) -> Dict[frozenset, List[Dict[str, Any]]]:
    m: Dict[frozenset, List[Dict[str, Any]]] = {}
    for p in constraints.get("close_pairs", []):
        l1 = p.get("label1")
        l2 = p.get("label2")
        if not isinstance(l1, str) or not isinstance(l2, str):
            continue
        key = frozenset([l1, l2])
        m.setdefault(key, []).append(p)
    return m


def _anchors_for_ordered_pair(pair_entry: Dict[str, Any], a: str, b: str) -> Tuple[np.ndarray, np.ndarray]:
    l1 = pair_entry["label1"]
    l2 = pair_entry["label2"]
    pts1 = np.array(pair_entry.get("anchors_label1_xy", []), dtype=np.float64)
    pts2 = np.array(pair_entry.get("anchors_label2_xy", []), dtype=np.float64)

    K = min(len(pts1), len(pts2))
    if K <= 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    if a == l1 and b == l2:
        return pts1[:K], pts2[:K]
    if a == l2 and b == l1:
        return pts2[:K], pts1[:K]

    return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)


def _compute_translation(parent_pts: np.ndarray, child_pts: np.ndarray) -> np.ndarray:
    K = min(parent_pts.shape[0], child_pts.shape[0])
    if K <= 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    diff = parent_pts[:K] - child_pts[:K]
    return np.mean(diff, axis=0).astype(np.float64)


def _apply_translation_to_points(pts: np.ndarray, t: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    return (pts + t[None, :]).astype(np.float64)


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
        borderValue=0
    )
    return ((out > 0).astype(np.uint8) * 255)


def _mean_l2(a: np.ndarray, b: np.ndarray) -> float:
    K = min(a.shape[0], b.shape[0])
    if K <= 0:
        return 0.0
    d = a[:K] - b[:K]
    return float(np.mean(np.sqrt(np.sum(d * d, axis=1))))


# ----------------------------
# Main
# ----------------------------
def main():
    if not os.path.exists(HIERARCHY_PATH):
        raise FileNotFoundError(f"Missing hierarchy: {HIERARCHY_PATH}")

    hierarchy = _load_json(HIERARCHY_PATH)
    if not isinstance(hierarchy, dict) or len(hierarchy) == 0:
        raise ValueError(f"hierarchy_tree.json must be a non-empty dict: {HIERARCHY_PATH}")

    roots = _find_roots(hierarchy)
    if len(roots) == 0:
        raise ValueError("No root found in hierarchy_tree.json (no node has parent=null).")
    root_label = sorted(roots)[0]

    for x in range(NUM_VIEWS):
        view_name = f"view_{x}"
        view_dir = os.path.join(OUT_ROOT, view_name)

        constraints_path = os.path.join(view_dir, "constraints", "close_boundaries_summary.json")
        homography_path = os.path.join(view_dir, "homography", "homography_results.json")

        if not os.path.exists(constraints_path) or not os.path.exists(homography_path):
            continue

        constraints = _load_json(constraints_path)
        homores = _load_json(homography_path)

        labels_info = homores.get("labels", {})
        H_img = int(homores.get("H", 0))
        W_img = int(homores.get("W", 0))
        if H_img <= 0 or W_img <= 0:
            continue

        # base image (for overlays)
        base_bgr = None
        img_rel = homores.get("image", None)
        if isinstance(img_rel, str):
            abs_img = os.path.join(ROOT, img_rel) if not os.path.isabs(img_rel) else img_rel
            base_bgr = cv2.imread(abs_img, cv2.IMREAD_COLOR)
        if base_bgr is None:
            base_bgr = np.zeros((H_img, W_img, 3), dtype=np.uint8)

        # Output dirs
        out_dir = os.path.join(view_dir, "paste_back")
        _ensure_dir(out_dir)
        masks_final_dir = os.path.join(out_dir, "masks_final")
        edges_dir = os.path.join(out_dir, "edges")
        _ensure_dir(masks_final_dir)
        _ensure_dir(edges_dir)

        with open(os.path.join(out_dir, "root_label.txt"), "w") as f:
            f.write(root_label + "\n")

        # Build anchor pair index
        pair_map = _collect_pair_anchors(constraints)

        # Helper to load a label's homography warped mask
        def load_warped(label: str) -> Optional[np.ndarray]:
            if label not in labels_info:
                return None
            outp = labels_info[label].get("outputs", {})
            rel = outp.get("mask_warped", None)
            if not isinstance(rel, str):
                return None
            abs_p = os.path.join(ROOT, rel) if not os.path.isabs(rel) else rel
            return _imread_mask_u8(abs_p, H_img, W_img)

        # Root initial mask (fixed)
        root_mask = load_warped(root_label)
        if root_mask is None:
            with open(os.path.join(out_dir, "paste_back_results_edges.json"), "w") as f:
                json.dump({
                    "view": view_name,
                    "root_label": root_label,
                    "status": "root_missing_in_homography",
                }, f, indent=2)
            continue

        final_masks: Dict[str, np.ndarray] = {root_label: root_mask}

        # BFS from root
        q: List[str] = [root_label]
        visited: Set[str] = set([root_label])

        edge_results: List[Dict[str, Any]] = []

        while q:
            parent = q.pop(0)
            for child in _children_of(hierarchy, parent):
                if child in visited:
                    continue
                visited.add(child)
                q.append(child)

                parent_mask = final_masks.get(parent, None)
                child_before = load_warped(child)
                if parent_mask is None or child_before is None:
                    edge_results.append({
                        "parent": parent,
                        "child": child,
                        "status": "missing_masks",
                        "has_parent_mask": parent_mask is not None,
                        "has_child_warped": child_before is not None,
                    })
                    continue

                key = frozenset([parent, child])
                pair_entries = pair_map.get(key, [])

                P_all = []
                C_all = []
                for pe in pair_entries:
                    P, C = _anchors_for_ordered_pair(pe, parent, child)
                    if P.shape[0] > 0:
                        P_all.append(P)
                        C_all.append(C)

                edge_dir = os.path.join(edges_dir, f"{parent}__{child}")
                _ensure_dir(edge_dir)

                if len(P_all) == 0:
                    child_after = child_before.copy()
                    final_masks[child] = child_after

                    cv2.imwrite(os.path.join(edge_dir, "child_before.png"), child_before)
                    cv2.imwrite(os.path.join(edge_dir, "child_after.png"), child_after)

                    ov_before = _overlay_mask(_overlay_mask(base_bgr, parent_mask, COL_PARENT_MASK, ALPHA, CONTOUR_THICK),
                                              child_before, COL_CHILD_BEFORE_MASK, ALPHA, CONTOUR_THICK)
                    ov_after = _overlay_mask(_overlay_mask(base_bgr, parent_mask, COL_PARENT_MASK, ALPHA, CONTOUR_THICK),
                                             child_after, COL_CHILD_AFTER_MASK, ALPHA, CONTOUR_THICK)
                    cv2.imwrite(os.path.join(edge_dir, "overlay_before.png"), ov_before)
                    cv2.imwrite(os.path.join(edge_dir, "overlay_after.png"), ov_after)

                    edge_results.append({
                        "parent": parent,
                        "child": child,
                        "status": "no_anchors_keep_as_is",
                        "edge_output_dir": os.path.relpath(edge_dir, out_dir),
                    })
                    continue

                parent_pts = np.concatenate(P_all, axis=0)
                child_pts = np.concatenate(C_all, axis=0)

                t = _compute_translation(parent_pts, child_pts)
                dx, dy = float(t[0]), float(t[1])

                child_after = _translate_mask(child_before, dx, dy)
                final_masks[child] = child_after

                err_before = _mean_l2(parent_pts, child_pts)
                child_pts_after = _apply_translation_to_points(child_pts, t)
                err_after = _mean_l2(parent_pts, child_pts_after)

                # Save masks
                cv2.imwrite(os.path.join(edge_dir, "child_before.png"), child_before)
                cv2.imwrite(os.path.join(edge_dir, "child_after.png"), child_after)

                # Mask overlays
                ov_before = _overlay_mask(_overlay_mask(base_bgr, parent_mask, COL_PARENT_MASK, ALPHA, CONTOUR_THICK),
                                          child_before, COL_CHILD_BEFORE_MASK, ALPHA, CONTOUR_THICK)
                ov_after = _overlay_mask(_overlay_mask(base_bgr, parent_mask, COL_PARENT_MASK, ALPHA, CONTOUR_THICK),
                                         child_after, COL_CHILD_AFTER_MASK, ALPHA, CONTOUR_THICK)
                cv2.imwrite(os.path.join(edge_dir, "overlay_before.png"), ov_before)
                cv2.imwrite(os.path.join(edge_dir, "overlay_after.png"), ov_after)

                # Anchor-fit vis (base image)
                vis_fit = base_bgr.copy()
                _draw_points(vis_fit, parent_pts, COL_ANCH_PARENT, radius=ANCHOR_RADIUS)
                _draw_points(vis_fit, child_pts, COL_ANCH_CHILD_BEFORE, radius=ANCHOR_RADIUS)
                _draw_points(vis_fit, child_pts_after, (0, 255, 255), radius=ANCHOR_RADIUS)  # yellow = child after
                cv2.imwrite(os.path.join(edge_dir, "anchor_fit_before_after.png"), vis_fit)

                vis_lines_before = base_bgr.copy()
                _draw_points(vis_lines_before, parent_pts, COL_ANCH_PARENT, radius=ANCHOR_RADIUS)
                _draw_points(vis_lines_before, child_pts, COL_ANCH_CHILD_BEFORE, radius=ANCHOR_RADIUS)
                _draw_match_lines(vis_lines_before, parent_pts, child_pts, COL_LINE, thickness=1)
                cv2.imwrite(os.path.join(edge_dir, "anchor_fit_lines_before.png"), vis_lines_before)

                vis_lines_after = base_bgr.copy()
                _draw_points(vis_lines_after, parent_pts, COL_ANCH_PARENT, radius=ANCHOR_RADIUS)
                _draw_points(vis_lines_after, child_pts_after, COL_ANCH_CHILD_AFTER, radius=ANCHOR_RADIUS)
                _draw_match_lines(vis_lines_after, parent_pts, child_pts_after, COL_LINE, thickness=1)
                cv2.imwrite(os.path.join(edge_dir, "anchor_fit_lines_after.png"), vis_lines_after)

                # ----------------------------
                # NEW: Anchors shown on BEFORE/AFTER mask overlays (so you see them move)
                # ----------------------------
                anchors_on_before = base_bgr.copy()
                anchors_on_before = _overlay_mask(anchors_on_before, parent_mask, COL_PARENT_MASK, alpha=ALPHA, contour_thick=1)
                anchors_on_before = _overlay_mask(anchors_on_before, child_before, COL_CHILD_BEFORE_MASK, alpha=ALPHA, contour_thick=1)
                _draw_points(anchors_on_before, parent_pts, COL_ANCH_PARENT, radius=ANCHOR_RADIUS + 1)
                _draw_points(anchors_on_before, child_pts, COL_ANCH_CHILD_BEFORE, radius=ANCHOR_RADIUS + 1)
                cv2.imwrite(os.path.join(edge_dir, "anchors_on_masks_before.png"), anchors_on_before)

                anchors_on_after = base_bgr.copy()
                anchors_on_after = _overlay_mask(anchors_on_after, parent_mask, COL_PARENT_MASK, alpha=ALPHA, contour_thick=1)
                anchors_on_after = _overlay_mask(anchors_on_after, child_after, COL_CHILD_AFTER_MASK, alpha=ALPHA, contour_thick=1)
                _draw_points(anchors_on_after, parent_pts, COL_ANCH_PARENT, radius=ANCHOR_RADIUS + 1)
                _draw_points(anchors_on_after, child_pts_after, COL_ANCH_CHILD_AFTER, radius=ANCHOR_RADIUS + 1)
                cv2.imwrite(os.path.join(edge_dir, "anchors_on_masks_after.png"), anchors_on_after)

                edge_results.append({
                    "parent": parent,
                    "child": child,
                    "status": "moved",
                    "translation_dxdy": [dx, dy],
                    "num_anchor_pairs_used": int(min(parent_pts.shape[0], child_pts.shape[0])),
                    "mean_l2_error_before": err_before,
                    "mean_l2_error_after": err_after,
                    "edge_output_dir": os.path.relpath(edge_dir, out_dir),
                })

        # Save final masks
        final_paths = {}
        for lbl, m in final_masks.items():
            p = os.path.join(masks_final_dir, f"{lbl}_mask_final.png")
            cv2.imwrite(p, m)
            final_paths[lbl] = os.path.relpath(p, ROOT)

        # Overall overlay of final masks
        overlay_all = base_bgr.copy()
        overlay_all = _overlay_mask(overlay_all, final_masks[root_label], COL_PARENT_MASK, alpha=ALPHA, contour_thick=1)
        for lbl, m in final_masks.items():
            if lbl == root_label:
                continue
            overlay_all = _overlay_mask(overlay_all, m, COL_CHILD_AFTER_MASK, alpha=ALPHA, contour_thick=1)

        overlay_path = os.path.join(out_dir, "all_final_masks_overlay.png")
        cv2.imwrite(overlay_path, overlay_all)

        out_json = {
            "view": view_name,
            "root_label": root_label,
            "status": "ok_edges_parent_fixed_child_translated",
            "constraints_json": os.path.relpath(constraints_path, ROOT),
            "homography_json": os.path.relpath(homography_path, ROOT),
            "reached_labels_from_root": sorted(list(final_masks.keys())),
            "edge_results": edge_results,
            "outputs": {
                "paste_back_dir": os.path.relpath(out_dir, ROOT),
                "masks_final_dir": os.path.relpath(masks_final_dir, ROOT),
                "edges_dir": os.path.relpath(os.path.join(out_dir, "edges"), ROOT),
                "all_final_masks_overlay": os.path.relpath(overlay_path, ROOT),
            },
            "final_masks": final_paths,
        }
        with open(os.path.join(out_dir, "paste_back_results_edges.json"), "w") as f:
            json.dump(out_json, f, indent=2)


if __name__ == "__main__":
    main()
