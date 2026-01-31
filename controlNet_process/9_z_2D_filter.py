#!/usr/bin/env python3
"""
Launcher: 2D filter attachments by aggregated 2D mask neighbors.

Now additionally:
- VIS deleted attachments first (graph_building/vis_deleted.py)
- VIS kept attachments after (graph_building/vis.py)

Assumptions:
- constraints json contains:
    - object_space
    - nodes[name].label_id
    - nodes[name].obb (used only for visualization bboxes in vis.py via bboxes_by_name)
- point cloud + assigned_ids are loaded from:
    sketch/3d_reconstruction/fused_model.ply
    sketch/3d_reconstruction/assigned_ids.npy   (or sketch/assigned_ids.npy fallback)
If your assigned_ids path differs, adjust ONLY the two path lines below.
"""

import os
import re
import json
import importlib.util
from collections import defaultdict
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image

import open3d as o3d

from graph_building.vis_deleted import verify_relations_vis_deleted
from graph_building.vis import verify_relations_vis


# ----------------------------
# Config
# ----------------------------
TOL_RATIO = 0.10
VIEWS = range(6)
FNAME_RE = re.compile(r"^(?P<label>.+)_(?P<idx>\d+)_mask\.png$", re.IGNORECASE)

VIS = False   # set to False to disable ALL visualization

# ----------------------------
# Utilities
# ----------------------------
def _pair(a: str, b: str):
    return tuple(sorted((a, b)))


def _is_unknown(name: str) -> bool:
    return isinstance(name, str) and name.startswith("unknown_")


def _load_module_from_path(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------
# Fallback aggregation (only used if graph_building/2D_relations_aggregation.py is missing)
# ----------------------------
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


def _compute_all_view_relations_fallback(seg_root: str):
    relations_all = set()
    views_found = 0
    total_masks = 0

    for x in VIEWS:
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
            mid = f"{label}_{idx}"
            items.append((mid, os.path.join(vd, f)))

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

                tol_px = int(np.ceil(TOL_RATIO * min(_bbox_diag(bb_a), _bbox_diag(bb_b))))
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
        "tol_ratio": TOL_RATIO,
        "used": "fallback_in_launcher",
    }
    return relations_all, stats


# ----------------------------
# Helpers for visualization inputs
# ----------------------------
def _load_pts_and_assigned_ids(root: str):
    """
    Load the exact visualization point cloud + assigned ids used in your label assignment pipeline.

    Paths (from your snippet):
      SAVE_DIR = ROOT/sketch/partfield_overlay/label_assignment_k20
      PLY_PATH = SAVE_DIR/assignment_colored.ply
      IDS_PATH = SAVE_DIR/assigned_label_ids.npy
    """
    save_dir = os.path.join(root, "sketch", "partfield_overlay", "label_assignment_k20")
    ply_path = os.path.join(save_dir, "assignment_colored.ply")
    ids_path = os.path.join(save_dir, "assigned_label_ids.npy")

    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")
    if not os.path.isfile(ids_path):
        raise FileNotFoundError(f"assigned ids not found: {ids_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)

    assigned_ids = np.load(ids_path)
    assigned_ids = np.asarray(assigned_ids).reshape(-1).astype(np.int32)

    if pts.shape[0] != assigned_ids.shape[0]:
        raise ValueError(f"pts N={pts.shape[0]} but assigned_ids N={assigned_ids.shape[0]}")

    return pts, assigned_ids, ply_path, ids_path


def _make_bboxes_by_name_from_constraints(data: dict):
    """
    graph_building/vis.py expects bboxes_by_name[name] to contain:
      - label_id
      - obb_pca: {center, axes, extents}
    Your constraints file stores nodes[name].obb, so we map it to obb_pca.
    """
    nodes = data.get("nodes", {})
    out = {}
    for name, entry in nodes.items():
        if not isinstance(entry, dict):
            continue
        lid = entry.get("label_id", None)
        obb = entry.get("obb", None)
        if lid is None or obb is None:
            continue
        if not all(k in obb for k in ["center", "axes", "extents"]):
            continue
        out[name] = {
            "label_id": int(lid),
            "obb_pca": {
                "center": obb["center"],
                "axes": obb["axes"],
                "extents": obb["extents"],
            },
        }
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    root = os.path.dirname(os.path.abspath(__file__))

    seg_root = os.path.join(root, "sketch", "segmentation_original_image")
    constraints_path = os.path.join(root, "sketch", "AEP", "initial_constraints.json")
    output_path = os.path.join(root, "sketch", "AEP", "filtered_relations.json")

    if not os.path.isdir(seg_root):
        raise FileNotFoundError(f"seg_root not found: {seg_root}")
    if not os.path.isfile(constraints_path):
        raise FileNotFoundError(f"constraints not found: {constraints_path}")

    # 1) Compute aggregated 2D relations using graph_building module if available
    agg_path = os.path.join(root, "graph_building", "2D_relations_aggregation.py")
    if os.path.isfile(agg_path):
        agg = _load_module_from_path(agg_path, "twoD_relations_aggregation")
        all_view_relations, rel_stats = agg.compute_all_view_relations(
            seg_root=seg_root,
            views=VIEWS,
            tol_ratio=TOL_RATIO,
        )
        rel_stats = dict(rel_stats)
        rel_stats["used"] = "graph_building/2D_relations_aggregation.py"
    else:
        all_view_relations, rel_stats = _compute_all_view_relations_fallback(seg_root)

    # 2) Read JSON, extract attachments
    with open(constraints_path, "r") as f:
        data = json.load(f)

    attachments = data.get("attachments", [])
    attach_pairs = set()
    attach_pair_to_count = defaultdict(int)

    for rel in attachments:
        a = rel.get("a")
        b = rel.get("b")
        if a and b:
            p = _pair(a, b)
            attach_pairs.add(p)
            attach_pair_to_count[p] += 1

    # 3) Determine missing pairs (JSON not in 2D), removable only if non-unknown
    missing_pairs_all = sorted([p for p in attach_pairs if p not in all_view_relations])

    missing_non_unknown = []
    for a, b in missing_pairs_all:
        if _is_unknown(a) or _is_unknown(b):
            continue
        missing_non_unknown.append((a, b))
    missing_non_unknown_set = set(missing_non_unknown)

    # 4) Filter attachment entries (never remove unknown-related)
    filtered_attachments = []
    deleted_attachments = []
    removed_entries = 0

    for rel in attachments:
        a = rel.get("a")
        b = rel.get("b")
        if not a or not b:
            filtered_attachments.append(rel)
            continue

        if _is_unknown(a) or _is_unknown(b):
            filtered_attachments.append(rel)
            continue

        if _pair(a, b) in missing_non_unknown_set:
            removed_entries += 1
            deleted_attachments.append(rel)
            continue

        filtered_attachments.append(rel)

    # 5) Save filtered json
    out_data = dict(data)
    out_data["attachments"] = filtered_attachments
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=2)

    # 6) Compute saved attachment unique pairs for printing
    saved_pairs = set()
    for rel in filtered_attachments:
        a = rel.get("a")
        b = rel.get("b")
        if a and b:
            saved_pairs.add(_pair(a, b))
    saved_pairs = sorted(saved_pairs)

    # 7) Clean prints
    print("=" * 100)
    print("2D FILTER (CLEAN)")
    print("-" * 100)
    print(f"seg_root:        {seg_root}")
    print(f"constraints_in:  {constraints_path}")
    print(f"filtered_out:    {output_path}")
    print(f"views:           {list(VIEWS)}")
    print(f"tol_ratio:       {TOL_RATIO}")
    print(f"2D rel source:   {rel_stats.get('used', 'unknown')}")
    print("-" * 100)
    print(f"2D relations (all views):       {rel_stats.get('relations_all_count', len(all_view_relations))}")
    print(f"JSON attachment raw entries:    {len(attachments)}")
    print(f"JSON attachment unique pairs:   {len(attach_pairs)}")
    print("-" * 100)
    print(f"Removable missing pairs (non-unknown): {len(missing_non_unknown)}")
    print(f"Attachment entries removed:           {removed_entries}")
    print(f"Attachment entries kept:              {len(filtered_attachments)}")
    print("-" * 100)

    if missing_non_unknown:
        print("Removed attachment pairs (unique, non-unknown):")
        for a, b in missing_non_unknown:
            print(f"  {a}  <->  {b}   (entries={attach_pair_to_count[_pair(a,b)]})")
    else:
        print("Removed attachment pairs (unique, non-unknown): (none)")

    print("-" * 100)
    print(f"Saved attachment relations (unique pairs): {len(saved_pairs)}")
    for a, b in saved_pairs:
        print(f"  {a}  <->  {b}")
    print("=" * 100)

    # ----------------------------
    # 8) VIS: deleted first, then kept
    # ----------------------------
    # Load geometry + ids for visualization
    pts, assigned_ids, ply_path, ids_path = _load_pts_and_assigned_ids(root)

    # Build bboxes_by_name in the format vis.py expects
    bboxes_by_name = _make_bboxes_by_name_from_constraints(data)

    obj_space = data.get("object_space", None)
    if obj_space is None:
        raise KeyError("constraints json missing key: object_space")

    symmetry = data.get("symmetry", {})
    containment = data.get("containment", [])

    # 8.1 VIS deleted relations
    if len(deleted_attachments) > 0 and VIS:
        print(f"[VIS] Deleted attachments entries: {len(deleted_attachments)} (showing first)")
        verify_relations_vis_deleted(
            pts=pts,
            assigned_ids=assigned_ids,
            bboxes_by_name=bboxes_by_name,
            symmetry=symmetry,
            attachments=deleted_attachments,
            object_space=obj_space,
            containment=containment,
            vis_anchor_points=True,
            anchor_radius=0.002,
            ignore_unknown=False,
        )
    else:
        print("[VIS] No deleted attachments to visualize.")

    # 8.2 VIS kept relations
    print(f"[VIS] Kept attachments entries: {len(filtered_attachments)} (showing after deleted)")
    if VIS:
        verify_relations_vis(
            pts=pts,
            assigned_ids=assigned_ids,
            bboxes_by_name=bboxes_by_name,
            symmetry=symmetry,
            attachments=filtered_attachments,
            object_space=obj_space,
            containment=containment,
            vis_anchor_points=True,
            anchor_radius=0.002,
            ignore_unknown=False,
        )


if __name__ == "__main__":
    main()
