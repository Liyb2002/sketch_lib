#!/usr/bin/env python3
# homography/move_anchor_points.py
#
# Visualize anchor movement under homography.
#
# Inputs:
#   sketch/back_project_masks/view_{x}/constraints/close_boundaries_summary.json
#   sketch/back_project_masks/view_{x}/homography/homography_results.json
#   sketch/segmentation_original_image/view_{x}/{label}_mask.png
#   sketch/views/view_{x}.png
#
# Outputs (per view):
#   sketch/back_project_masks/view_{x}/moved_anchor/
#     - moved_anchor_all.png                 (SIDE-BY-SIDE BEFORE|AFTER + lines)
#     - pairs/{label1}__{label2}.png         (SIDE-BY-SIDE BEFORE|AFTER + lines)
#     - moved_anchor_points.json

import os
import json
import shutil
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIEWS_DIR = os.path.join(ROOT, "sketch", "views")
SEG_DIR = os.path.join(ROOT, "sketch", "segmentation_original_image")
OUT_ROOT = os.path.join(ROOT, "sketch", "back_project_masks")

NUM_VIEWS = 6

# Overlay style
ALPHA_FILL = 0.22
CONTOUR_THICK = 2

# Anchor drawing
R_ANCHOR = 2
LINE_THICK = 1

# Pair colors (BGR)
C_MASK_L1 = (255, 0, 0)   # blue mask
C_MASK_L2 = (0, 0, 255)   # red mask
C_L1 = (255, 255, 0)      # cyan anchors (label1)
C_L2 = (255, 0, 255)      # magenta anchors (label2)
C_LINE = (0, 255, 255)    # yellow lines


def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _clean_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _ensure_float_pts(pts_xy) -> np.ndarray:
    arr = np.array(pts_xy, dtype=np.float32) if pts_xy is not None else np.zeros((0, 2), np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return arr.reshape(-1, 2).astype(np.float32)


def _apply_homography_points(pts_xy: np.ndarray, Hmat: np.ndarray) -> np.ndarray:
    if pts_xy is None or pts_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    Hmat = np.asarray(Hmat, dtype=np.float64)
    pts = pts_xy.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, Hmat).reshape(-1, 2).astype(np.float32)
    return out


def _draw_points(img, pts_xy, color, r=2):
    for x, y in pts_xy:
        cv2.circle(img, (int(round(x)), int(round(y))), r, color, thickness=-1)


def _overlay_mask_on_image(img_bgr, mask_u8, color_bgr, alpha=0.22, contour_thick=2):
    out = img_bgr.copy()
    if mask_u8 is None:
        return out
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


def _imread_mask_gray(path: str, H_img: int, W_img: int):
    if not path or (not os.path.exists(path)):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[:2] != (H_img, W_img):
        m = cv2.resize(m, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
    m = (m > 0).astype(np.uint8) * 255
    return m


def _nn_lines(src_xy: np.ndarray, dst_xy: np.ndarray):
    """
    Build NN pairings for visualization: for each src point, connect to nearest dst point.
    Returns list of (p_src, p_dst).
    """
    if src_xy is None or dst_xy is None:
        return []
    if src_xy.size == 0 or dst_xy.size == 0:
        return []

    # (Ns, 2) vs (Nd, 2)
    s = src_xy.astype(np.float32)
    d = dst_xy.astype(np.float32)

    # squared distances (Ns, Nd)
    ds = s[:, None, :] - d[None, :, :]
    dist2 = ds[..., 0] * ds[..., 0] + ds[..., 1] * ds[..., 1]
    nn_idx = np.argmin(dist2, axis=1)

    pairs = []
    for i in range(s.shape[0]):
        pairs.append((s[i], d[nn_idx[i]]))
    return pairs


def _draw_lines(img, pairs, color, thick=1):
    for p, q in pairs:
        x1, y1 = int(round(float(p[0]))), int(round(float(p[1])))
        x2, y2 = int(round(float(q[0]))), int(round(float(q[1])))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness=thick, lineType=cv2.LINE_AA)


def main():
    for x in range(NUM_VIEWS):
        view_name = f"view_{x}"

        img_path = os.path.join(VIEWS_DIR, f"{view_name}.png")
        constraints_json = os.path.join(OUT_ROOT, view_name, "constraints", "close_boundaries_summary.json")
        homography_json = os.path.join(OUT_ROOT, view_name, "homography", "homography_results.json")

        if not (os.path.exists(img_path) and os.path.exists(constraints_json) and os.path.exists(homography_json)):
            continue

        base = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if base is None:
            continue
        H_img, W_img = base.shape[:2]

        cons = _read_json(constraints_json)
        homo = _read_json(homography_json)

        close_pairs = cons.get("close_pairs", [])
        label_centroids = cons.get("label_centroids_xy", {})  # REQUIRED for relative anchors
        homo_labels = (homo.get("labels") or {})

        # Output folder (CLEAN)
        out_view = os.path.join(OUT_ROOT, view_name, "moved_anchor")
        _clean_dir(out_view)
        out_pairs = os.path.join(out_view, "pairs")
        os.makedirs(out_pairs, exist_ok=True)

        def _seg_mask_path(label: str) -> str:
            return os.path.join(SEG_DIR, view_name, f"{label}_mask.png")

        def _warped_mask_path(label: str) -> str:
            ent = homo_labels.get(label, None)
            if ent is None:
                return ""
            outp = ent.get("outputs", {}) or {}
            rel = outp.get("mask_warped", None)
            if not rel:
                return ""
            return os.path.join(ROOT, rel)

        def _H_for_label(label: str) -> np.ndarray:
            ent = homo_labels.get(label, None)
            if ent is None:
                return np.eye(3, dtype=np.float64)
            Hlist = ent.get("homography_before_to_after", None)
            if Hlist is None:
                return np.eye(3, dtype=np.float64)
            Hmat = np.array(Hlist, dtype=np.float64)
            if Hmat.shape != (3, 3):
                return np.eye(3, dtype=np.float64)
            return Hmat

        moved_pairs_json = []

        # For moved_anchor_all.png (aggregate)
        agg_before = base.copy()
        agg_after = base.copy()

        for p in close_pairs:
            l1 = p.get("label1")
            l2 = p.get("label2")
            if l1 is None or l2 is None:
                continue

            c1 = label_centroids.get(l1, None)
            c2 = label_centroids.get(l2, None)
            if c1 is None or c2 is None:
                # If you haven't added centroids to constraints.json yet, we can't convert relative->absolute.
                continue

            # relative anchors (dxy) + centroid -> absolute BEFORE anchors
            a1_dxy = _ensure_float_pts(p.get("anchors_label1_dxy", []))
            a2_dxy = _ensure_float_pts(p.get("anchors_label2_dxy", []))

            c1_xy = np.array(c1, dtype=np.float32).reshape(1, 2)
            c2_xy = np.array(c2, dtype=np.float32).reshape(1, 2)

            a1_before = (a1_dxy + c1_xy).astype(np.float32)
            a2_before = (a2_dxy + c2_xy).astype(np.float32)

            # AFTER anchors: apply homography
            H1 = _H_for_label(l1)
            H2 = _H_for_label(l2)
            a1_after = _apply_homography_points(a1_before, H1)
            a2_after = _apply_homography_points(a2_before, H2)

            # BEFORE masks
            m1_before = _imread_mask_gray(_seg_mask_path(l1), H_img, W_img)
            m2_before = _imread_mask_gray(_seg_mask_path(l2), H_img, W_img)

            # AFTER masks
            m1_after = _imread_mask_gray(_warped_mask_path(l1), H_img, W_img)
            m2_after = _imread_mask_gray(_warped_mask_path(l2), H_img, W_img)

            # Build before/after vis for this pair
            vis_before = base.copy()
            vis_after = base.copy()

            vis_before = _overlay_mask_on_image(vis_before, m1_before, C_MASK_L1, alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK)
            vis_before = _overlay_mask_on_image(vis_before, m2_before, C_MASK_L2, alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK)

            vis_after = _overlay_mask_on_image(vis_after, m1_after, C_MASK_L1, alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK)
            vis_after = _overlay_mask_on_image(vis_after, m2_after, C_MASK_L2, alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK)

            # Lines (NN for visualization only) + anchors
            lines_before = _nn_lines(a1_before, a2_before)
            lines_after = _nn_lines(a1_after, a2_after)

            _draw_lines(vis_before, lines_before, C_LINE, thick=LINE_THICK)
            _draw_lines(vis_after, lines_after, C_LINE, thick=LINE_THICK)

            _draw_points(vis_before, a1_before, C_L1, r=R_ANCHOR)
            _draw_points(vis_before, a2_before, C_L2, r=R_ANCHOR)

            _draw_points(vis_after, a1_after, C_L1, r=R_ANCHOR)
            _draw_points(vis_after, a2_after, C_L2, r=R_ANCHOR)

            # Side-by-side
            side = np.concatenate([vis_before, vis_after], axis=1)
            cv2.putText(side, "BEFORE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(side, "AFTER", (W_img + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            out_pair = os.path.join(out_pairs, f"{l1}__{l2}.png")
            cv2.imwrite(out_pair, side)

            moved_pairs_json.append({
                "label1": l1,
                "label2": l2,
                "anchors_label1_before_xy": a1_before.astype(float).tolist(),
                "anchors_label2_before_xy": a2_before.astype(float).tolist(),
                "anchors_label1_after_xy": a1_after.astype(float).tolist(),
                "anchors_label2_after_xy": a2_after.astype(float).tolist(),
                "pair_vis": os.path.relpath(out_pair, out_view),
            })

            # Aggregate (for moved_anchor_all.png):
            # Overlay masks + draw this pair’s anchors + lines onto agg_before/agg_after.
            agg_before = _overlay_mask_on_image(agg_before, m1_before, C_MASK_L1, alpha=ALPHA_FILL, contour_thick=1)
            agg_before = _overlay_mask_on_image(agg_before, m2_before, C_MASK_L2, alpha=ALPHA_FILL, contour_thick=1)
            _draw_lines(agg_before, lines_before, C_LINE, thick=1)
            _draw_points(agg_before, a1_before, C_L1, r=R_ANCHOR)
            _draw_points(agg_before, a2_before, C_L2, r=R_ANCHOR)

            agg_after = _overlay_mask_on_image(agg_after, m1_after, C_MASK_L1, alpha=ALPHA_FILL, contour_thick=1)
            agg_after = _overlay_mask_on_image(agg_after, m2_after, C_MASK_L2, alpha=ALPHA_FILL, contour_thick=1)
            _draw_lines(agg_after, lines_after, C_LINE, thick=1)
            _draw_points(agg_after, a1_after, C_L1, r=R_ANCHOR)
            _draw_points(agg_after, a2_after, C_L2, r=R_ANCHOR)

        # moved_anchor_all.png: SIDE-BY-SIDE
        side_all = np.concatenate([agg_before, agg_after], axis=1)
        cv2.putText(side_all, "BEFORE (all pairs)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side_all, "AFTER (all pairs)", (W_img + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        out_global = os.path.join(out_view, "moved_anchor_all.png")
        cv2.imwrite(out_global, side_all)

        out_json = {
            "view": view_name,
            "image": os.path.relpath(img_path, ROOT),
            "constraints_summary": os.path.relpath(constraints_json, ROOT),
            "homography_results": os.path.relpath(homography_json, ROOT),
            "pairs": moved_pairs_json,
            "outputs": {
                "pairs_dir": "pairs/",
                "moved_anchor_all": "moved_anchor_all.png",
            },
        }
        with open(os.path.join(out_view, "moved_anchor_points.json"), "w") as f:
            json.dump(out_json, f, indent=2)


if __name__ == "__main__":
    main()
