#!/usr/bin/env python3
# homography/paste_back.py
#
# Paste-back / translate warped masks to satisfy close-boundary anchor constraints,
# using the already-computed moved anchor locations under homography.
#
# Core idea (tree / root anchored):
#   - Root label is fixed (translation = [0,0]).
#   - For each parent->child edge in hierarchy_tree.json (FULL TREE):
#       * If anchor constraints exist in moved_anchor_points.json for (parent, child):
#           compute a TRANSLATION for the child (in image pixels) so that
#           child's moved anchors align to parent's moved anchors,
#           with parent already in its final translated position.
#       * If NO anchor constraints exist for (parent, child):
#           still propagate hierarchy by setting child's translation = parent's translation.
#           (So the entire hierarchy receives a well-defined translation and moves together.)
#   - Apply translations to warped masks (output of homography/homography.py).
#
# Inputs:
#   sketch/AEP/hierarchy_tree.json                 (dict: label -> {parent, children})
#   sketch/back_project_masks/view_{x}/homography/homography_results.json
#   sketch/back_project_masks/view_{x}/moved_anchor/moved_anchor_points.json
#   sketch/views/view_{x}.png
#
# Outputs:
#   sketch/back_project_masks/view_{x}/paste_back/
#     - before_overlay.png
#     - after_overlay.png
#     - masks/{label}_mask.png
#     - edges/{parent}__{child}.png                (side-by-side BEFORE|AFTER anchor fit; only when anchors exist)
#     - paste_back_results.json

import os
import json
import shutil
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HIER_PATH = os.path.join(ROOT, "sketch", "AEP", "hierarchy_tree.json")
VIEWS_DIR = os.path.join(ROOT, "sketch", "views")
OUT_ROOT  = os.path.join(ROOT, "sketch", "back_project_masks")

NUM_VIEWS = 6

# Visual style
ALPHA_FILL = 0.6
CONTOUR_THICK = 2

C_MASK_PARENT = (255, 0, 0)   # blue
C_MASK_CHILD  = (0, 0, 255)   # red
C_PARENT_ANCH = (255, 0, 0)   # cyan-ish (BGR)
C_CHILD_ANCH  = (255, 0, 255) # magenta
C_LINE        = (0, 255, 255) # yellow

R_ANCHOR = 3
LINE_THICK = 2


# ----------------------------
# IO utils
# ----------------------------
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


def _overlay_mask_on_image(img_bgr, mask_u8, color_bgr, alpha=0.25, contour_thick=2):
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


def _draw_points(img, pts_xy, color, r=2):
    for x, y in pts_xy:
        cv2.circle(img, (int(round(float(x))), int(round(float(y)))), r, color, thickness=-1)


def _draw_lines(img, pairs, color, thick=1):
    for p, q in pairs:
        x1, y1 = int(round(float(p[0]))), int(round(float(p[1])))
        x2, y2 = int(round(float(q[0]))), int(round(float(q[1])))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness=thick, lineType=cv2.LINE_AA)


def _nn_pairs(src_xy: np.ndarray, dst_xy: np.ndarray):
    """
    Visual-only nearest-neighbor pairing:
      for each src point, connect to nearest dst point.
    Returns list[(src, dst)].
    """
    if src_xy is None or dst_xy is None:
        return []
    if src_xy.size == 0 or dst_xy.size == 0:
        return []

    s = src_xy.astype(np.float32)
    d = dst_xy.astype(np.float32)

    diff = s[:, None, :] - d[None, :, :]
    dist2 = diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1]
    nn = np.argmin(dist2, axis=1)

    out = []
    for i in range(s.shape[0]):
        out.append((s[i], d[nn[i]]))
    return out


def _nn_stats(src_xy: np.ndarray, dst_xy: np.ndarray):
    """Return mean/median NN distance from src->dst (src anchors to nearest dst anchor)."""
    if src_xy is None or dst_xy is None or src_xy.size == 0 or dst_xy.size == 0:
        return {"mean": None, "median": None, "count": 0}

    s = src_xy.astype(np.float32)
    d = dst_xy.astype(np.float32)
    diff = s[:, None, :] - d[None, :, :]
    dist2 = diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1]
    nn = np.argmin(dist2, axis=1)
    nn_dist = np.sqrt(dist2[np.arange(s.shape[0]), nn])

    return {
        "mean": float(np.mean(nn_dist)) if nn_dist.size else None,
        "median": float(np.median(nn_dist)) if nn_dist.size else None,
        "count": int(nn_dist.size),
    }


def _robust_translation(parent_xy: np.ndarray, child_xy: np.ndarray):
    """
    Compute translation delta to apply to child so it aligns to parent.
    Uses mutual NN-ish robustness:
      - child->parent NN vectors
      - parent->child NN vectors (negated)
      - concatenate and take median vector
    """
    if parent_xy is None or child_xy is None or parent_xy.size == 0 or child_xy.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32), {"used": 0}

    p = parent_xy.astype(np.float32)
    c = child_xy.astype(np.float32)

    # child -> parent
    diff_cp = c[:, None, :] - p[None, :, :]
    dist2_cp = diff_cp[..., 0] * diff_cp[..., 0] + diff_cp[..., 1] * diff_cp[..., 1]
    nn_p = np.argmin(dist2_cp, axis=1)
    vec_cp = p[nn_p] - c  # vectors that move c to p

    # parent -> child
    diff_pc = p[:, None, :] - c[None, :, :]
    dist2_pc = diff_pc[..., 0] * diff_pc[..., 0] + diff_pc[..., 1] * diff_pc[..., 1]
    nn_c = np.argmin(dist2_pc, axis=1)
    vec_pc = p - c[nn_c]  # vectors that move (nearest child) to each parent

    vec = np.concatenate([vec_cp, vec_pc], axis=0)
    delta = np.median(vec, axis=0).astype(np.float32)

    return delta, {"used": int(vec.shape[0])}


def _warp_translate_mask(mask_u8: np.ndarray, dx: float, dy: float, out_w: int, out_h: int):
    if mask_u8 is None:
        return None
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    moved = cv2.warpAffine(
        mask_u8,
        M,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    moved = (moved > 0).astype(np.uint8) * 255
    return moved


# ----------------------------
# Hierarchy parsing
# ----------------------------
def _load_hierarchy_tree_dict(path: str):
    """
    Expects:
    {
      "wheel_0": {"parent": null, "children": [...]},
      ...
    }
    Returns: (tree_dict, root_label, edges_list[parent,child])
    """
    if not os.path.exists(path):
        return {}, None, []

    tree = _read_json(path)
    if not isinstance(tree, dict) or len(tree) == 0:
        return {}, None, []

    roots = [k for k, v in tree.items() if (isinstance(v, dict) and v.get("parent", None) is None)]
    root = roots[0] if roots else None

    edges = []
    for parent, info in tree.items():
        if not isinstance(info, dict):
            continue
        children = info.get("children", []) or []
        for ch in children:
            edges.append((parent, ch))

    return tree, root, edges


def _bfs_order(tree: dict, root: str):
    """Return BFS order list of nodes and edges (parent->child)."""
    if root is None or root not in tree:
        return [root] if root else [], []

    q = [root]
    seen = {root}
    order = []
    edges = []

    while q:
        u = q.pop(0)
        order.append(u)
        children = (tree.get(u, {}) or {}).get("children", []) or []
        for v in children:
            edges.append((u, v))
            if v not in seen:
                seen.add(v)
                q.append(v)
    return order, edges


# ----------------------------
# Anchor lookup from moved_anchor_points.json
# ----------------------------
def _build_pair_anchor_lookup(moved_anchor_json: dict):
    """
    Build mapping:
      frozenset({a,b}) -> record with anchors for a and b in AFTER coords
    Each record:
      {
        "a": labelA, "b": labelB,
        "a_after": (Na,2), "b_after": (Nb,2),
        "a_before": ..., "b_before": ...
      }
    """
    out = {}
    for rec in (moved_anchor_json.get("pairs") or []):
        l1 = rec.get("label1")
        l2 = rec.get("label2")
        if not l1 or not l2:
            continue

        a1b = _ensure_float_pts(rec.get("anchors_label1_before_xy", []))
        a2b = _ensure_float_pts(rec.get("anchors_label2_before_xy", []))
        a1a = _ensure_float_pts(rec.get("anchors_label1_after_xy", []))
        a2a = _ensure_float_pts(rec.get("anchors_label2_after_xy", []))

        out[frozenset([l1, l2])] = {
            "a": l1, "b": l2,
            "a_before": a1b, "b_before": a2b,
            "a_after": a1a, "b_after": a2a,
        }
    return out


def _get_pair_anchors(pair_map: dict, u: str, v: str):
    """
    Return (u_after, v_after, u_before, v_before) using stored orientation.
    """
    key = frozenset([u, v])
    rec = pair_map.get(key, None)
    if rec is None:
        return None, None, None, None

    if rec["a"] == u and rec["b"] == v:
        return rec["a_after"], rec["b_after"], rec["a_before"], rec["b_before"]
    if rec["a"] == v and rec["b"] == u:
        # swapped
        return rec["b_after"], rec["a_after"], rec["b_before"], rec["a_before"]

    return None, None, None, None


# ----------------------------
# Main
# ----------------------------
def main():
    tree, tree_root_label, _ = _load_hierarchy_tree_dict(HIER_PATH)

    for x in range(NUM_VIEWS):
        view_name = f"view_{x}"
        img_path = os.path.join(VIEWS_DIR, f"{view_name}.png")

        homo_json = os.path.join(OUT_ROOT, view_name, "homography", "homography_results.json")
        moved_json = os.path.join(OUT_ROOT, view_name, "moved_anchor", "moved_anchor_points.json")

        if not (os.path.exists(img_path) and os.path.exists(homo_json) and os.path.exists(moved_json)):
            continue

        base = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if base is None:
            continue
        H_img, W_img = base.shape[:2]

        homo = _read_json(homo_json)
        moved = _read_json(moved_json)
        pair_map = _build_pair_anchor_lookup(moved)

        homo_labels = (homo.get("labels") or {})

        # Always use hierarchy root if hierarchy exists; otherwise fallback to first moved pair label.
        root = tree_root_label if (tree and tree_root_label in tree) else None
        if root is None:
            mpairs = moved.get("pairs") or []
            root = mpairs[0].get("label1") if mpairs else None
        if root is None:
            continue

        # FULL hierarchy traversal (if hierarchy exists); otherwise no edges.
        hierarchy_used = bool(tree and root in tree)
        order_nodes, order_edges = _bfs_order(tree, root) if hierarchy_used else ([root], [])

        # FULL labels_involved = entire tree (when available) so every node gets a translation.
        labels_involved = set()
        if hierarchy_used:
            labels_involved.update(tree.keys())
        else:
            labels_involved.add(root)
            # if no hierarchy, still include labels appearing in moved pairs so we can write outputs
            for rec in moved.get("pairs") or []:
                a = rec.get("label1")
                b = rec.get("label2")
                if a:
                    labels_involved.add(a)
                if b:
                    labels_involved.add(b)

        # Build warped mask paths for labels
        def _warped_mask_path(label: str):
            ent = homo_labels.get(label, None)
            if ent is None:
                return None
            outp = ent.get("outputs", {}) or {}
            rel = outp.get("mask_warped", None)
            if not rel:
                return None
            return os.path.join(ROOT, rel)

        # Load warped masks for all labels we might touch (only those present in homography outputs)
        warped_masks = {}
        for lbl in sorted(labels_involved):
            p = _warped_mask_path(lbl)
            if p and os.path.exists(p):
                m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape[:2] != (H_img, W_img):
                        m = cv2.resize(m, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
                    warped_masks[lbl] = (m > 0).astype(np.uint8) * 255

        # Output dirs (CLEAN)
        out_dir = os.path.join(OUT_ROOT, view_name, "paste_back")
        _clean_dir(out_dir)
        out_masks = os.path.join(out_dir, "masks")
        out_edges = os.path.join(out_dir, "edges")
        os.makedirs(out_masks, exist_ok=True)
        os.makedirs(out_edges, exist_ok=True)

        # Current translations (accumulated) for each label
        t = {lbl: np.array([0.0, 0.0], dtype=np.float32) for lbl in labels_involved}
        t[root] = np.array([0.0, 0.0], dtype=np.float32)

        edge_reports = []

        # Process edges in BFS order: parent already fixed, child moves
        for parent, child in order_edges:
            if parent not in labels_involved or child not in labels_involved:
                continue

            # Default: if no anchors exist, propagate hierarchy by inheriting parent's translation.
            # (This ensures the FULL tree is assigned translations and moves consistently.)
            p_after, c_after, p_before, c_before = _get_pair_anchors(pair_map, parent, child)
            if p_after is None or c_after is None:
                t[child] = t[parent].copy()
                edge_reports.append({
                    "parent": parent,
                    "child": child,
                    "status": "no_pair_anchors_propagate_parent",
                    "child_translation_xy": [float(t[child][0]), float(t[child][1])],
                    "delta_child_xy": [0.0, 0.0],
                    "nn_dist_before": {"mean": None, "median": None, "count": 0},
                    "nn_dist_after": {"mean": None, "median": None, "count": 0},
                    "pair_vis": None,
                    "dbg": {"used": 0},
                })
                continue

            # anchors BEFORE translation application (current state = after homography + current t)
            p_curr = p_after + t[parent][None, :]
            c_curr = c_after + t[child][None, :]

            # compute child delta to align to parent
            delta, dbg = _robust_translation(p_curr, c_curr)

            # stats before applying delta
            stats_before = _nn_stats(c_curr, p_curr)

            # update child translation
            t[child] = t[child] + delta

            # anchors AFTER translation
            c_new = c_after + t[child][None, :]
            p_new = p_after + t[parent][None, :]

            stats_after = _nn_stats(c_new, p_new)

            # write per-edge visualization (side-by-side BEFORE | AFTER)
            visL = base.copy()
            visR = base.copy()

            m_parent = warped_masks.get(parent, None)
            m_child  = warped_masks.get(child, None)

            # BEFORE: show current before applying delta:
            t_child_before = t[child] - delta
            m_child_before = _warp_translate_mask(
                m_child, float(t_child_before[0]), float(t_child_before[1]), W_img, H_img
            ) if m_child is not None else None
            m_parent_vis = _warp_translate_mask(
                m_parent, float(t[parent][0]), float(t[parent][1]), W_img, H_img
            ) if m_parent is not None else None

            visL = _overlay_mask_on_image(visL, m_parent_vis, C_MASK_PARENT, alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK)
            visL = _overlay_mask_on_image(visL, m_child_before, C_MASK_CHILD,  alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK)

            lines_before = _nn_pairs(c_curr, p_curr)
            _draw_lines(visL, lines_before, C_LINE, thick=LINE_THICK)
            _draw_points(visL, p_curr, C_PARENT_ANCH, r=R_ANCHOR)
            _draw_points(visL, c_curr, C_CHILD_ANCH,  r=R_ANCHOR)

            # AFTER: child updated
            m_child_after = _warp_translate_mask(
                m_child, float(t[child][0]), float(t[child][1]), W_img, H_img
            ) if m_child is not None else None

            visR = _overlay_mask_on_image(visR, m_parent_vis, C_MASK_PARENT, alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK)
            visR = _overlay_mask_on_image(visR, m_child_after, C_MASK_CHILD,  alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK)

            lines_after = _nn_pairs(c_new, p_new)
            _draw_lines(visR, lines_after, C_LINE, thick=LINE_THICK)
            _draw_points(visR, p_new, C_PARENT_ANCH, r=R_ANCHOR)
            _draw_points(visR, c_new, C_CHILD_ANCH,  r=R_ANCHOR)

            side = np.concatenate([visL, visR], axis=1)
            cv2.putText(side, f"BEFORE  {parent}->{child}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(side, f"AFTER   {parent}->{child}", (W_img + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if stats_before["mean"] is not None and stats_after["mean"] is not None:
                txt = f"NN mean px: {stats_before['mean']:.2f} -> {stats_after['mean']:.2f}"
                cv2.putText(side, txt, (10, H_img - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(side, txt, (10, H_img - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            out_edge = os.path.join(out_edges, f"{parent}__{child}.png")
            cv2.imwrite(out_edge, side)

            edge_reports.append({
                "parent": parent,
                "child": child,
                "status": "paired_anchor_fit",
                "delta_child_xy": [float(delta[0]), float(delta[1])],
                "child_translation_xy": [float(t[child][0]), float(t[child][1])],
                "nn_dist_before": stats_before,
                "nn_dist_after": stats_after,
                "pair_vis": os.path.relpath(out_edge, out_dir),
                "dbg": dbg,
            })

        # Apply final translations to all loaded warped masks and save
        translated_masks = {}
        for lbl, m in warped_masks.items():
            moved_m = _warp_translate_mask(m, float(t[lbl][0]), float(t[lbl][1]), W_img, H_img)
            translated_masks[lbl] = moved_m
            cv2.imwrite(os.path.join(out_masks, f"{lbl}_mask.png"), moved_m)

        # Overlays
        before_overlay = base.copy()
        after_overlay  = base.copy()

        for lbl in sorted(labels_involved):
            m0 = warped_masks.get(lbl, None)
            m1 = translated_masks.get(lbl, None)
            if m0 is not None:
                before_overlay = _overlay_mask_on_image(before_overlay, m0, (0, 255, 0), alpha=0.12, contour_thick=1)
            if m1 is not None:
                after_overlay = _overlay_mask_on_image(after_overlay, m1, (0, 255, 0), alpha=0.12, contour_thick=1)

        cv2.imwrite(os.path.join(out_dir, "before_overlay.png"), before_overlay)
        cv2.imwrite(os.path.join(out_dir, "after_overlay.png"), after_overlay)

        # Save results JSON
        results = {
            "view": view_name,
            "root_label": root,
            "hierarchy_used": hierarchy_used,
            "labels_involved": sorted([l for l in labels_involved if l]),
            "translations_xy": {lbl: [float(t[lbl][0]), float(t[lbl][1])] for lbl in labels_involved if lbl},
            "edges_processed": edge_reports,
            "inputs": {
                "hierarchy_tree": os.path.relpath(HIER_PATH, ROOT) if os.path.exists(HIER_PATH) else None,
                "homography_results": os.path.relpath(homo_json, ROOT),
                "moved_anchor_points": os.path.relpath(moved_json, ROOT),
            },
            "outputs": {
                "before_overlay": "before_overlay.png",
                "after_overlay": "after_overlay.png",
                "masks_dir": "masks/",
                "edges_dir": "edges/",
            },
        }
        with open(os.path.join(out_dir, "paste_back_results.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
