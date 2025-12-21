#!/usr/bin/env python3
import os
import json
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import open3d as o3d


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _load_primitives(primitives_json_path: str) -> List[Dict[str, Any]]:
    data = _load_json(primitives_json_path)
    if isinstance(data, dict) and "primitives" in data:
        return data["primitives"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected primitives JSON format: {primitives_json_path}")


def _norm_label(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _obb_from_part(part: Dict[str, Any]) -> o3d.geometry.OrientedBoundingBox:
    params = part.get("parameters", {})
    center = np.array(params.get("center", [0, 0, 0]), dtype=np.float64)
    extent = np.array(params.get("extent", [0, 0, 0]), dtype=np.float64)
    rotation = np.array(params.get("rotation", np.eye(3)), dtype=np.float64)
    return o3d.geometry.OrientedBoundingBox(center, rotation, extent)


def _get_params(part: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = part.get("parameters", {})
    center = np.array(params.get("center", [0, 0, 0]), dtype=np.float64)
    extent = np.array(params.get("extent", [0, 0, 0]), dtype=np.float64)
    rotation = np.array(params.get("rotation", np.eye(3)), dtype=np.float64)
    return center, extent, rotation


def _infer_cluster_id_artifacts(ply_path: str) -> Tuple[str, str]:
    """
    Auto infer:
      - per-point cluster ids
      - cluster -> label json
    merged:
      <dsl_optimize>/merged_labeled_clusters.ply
      <dsl_optimize>/merged_cluster_ids.npy
      <dsl_optimize>/merged_cluster_to_label.json
    base:
      <clusters>/labeled_clusters.ply
      <clusters>/final_cluster_ids.npy
      <clusters>/cluster_to_label.json
    """
    ply_dir = os.path.dirname(os.path.abspath(ply_path))
    ply_base = os.path.basename(ply_path)

    if "merged_labeled_clusters.ply" in ply_base:
        return (
            os.path.join(ply_dir, "merged_cluster_ids.npy"),
            os.path.join(ply_dir, "merged_cluster_to_label.json"),
        )

    return (
        os.path.join(ply_dir, "final_cluster_ids.npy"),
        os.path.join(ply_dir, "cluster_to_label.json"),
    )


def _build_point_label_array(cluster_ids: np.ndarray, cluster_map_json: Dict[str, Any]) -> np.ndarray:
    cid_to_label: Dict[int, str] = {}
    for k, v in cluster_map_json.items():
        try:
            cid = int(k)
        except Exception:
            continue
        cid_to_label[cid] = _norm_label(v.get("label", "unknown"))

    out = np.empty((cluster_ids.shape[0],), dtype=object)
    out[:] = "unknown"
    for i, cid in enumerate(cluster_ids):
        cid = int(cid)
        if cid < 0:
            continue
        out[i] = cid_to_label.get(cid, "unknown")
    return out


def _format_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x):.6f}" for x in v.tolist()) + "]"


def run_before_after_vis_per_label(
    before_primitives_json: str,
    after_primitives_json: str,
    ply_path: str,
    *,
    # display-only: nudge after boxes so you ALWAYS see two boxes
    display_nudge_frac: float = 0.015,  # 1.5% of box diagonal
    display_min_nudge: float = 1e-4,    # absolute minimum
):
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    before = _load_primitives(before_primitives_json)
    after = _load_primitives(after_primitives_json)

    b_by_cid = {int(p.get("cluster_id", -1)): p for p in before}
    a_by_cid = {int(p.get("cluster_id", -1)): p for p in after}
    common_cids = sorted(set(b_by_cid.keys()).intersection(set(a_by_cid.keys())))

    if not common_cids:
        print("[VIS] No matching cluster_id between before/after.")
        return

    print(f"[VIS] Loading PLY: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None

    cluster_ids_path, cluster_map_path = _infer_cluster_id_artifacts(ply_path)
    if not os.path.exists(cluster_ids_path):
        raise FileNotFoundError(f"Missing cluster ids npy for visualization: {cluster_ids_path}")
    if not os.path.exists(cluster_map_path):
        raise FileNotFoundError(f"Missing cluster_to_label json for visualization: {cluster_map_path}")

    cluster_ids = np.load(cluster_ids_path).reshape(-1)
    if cluster_ids.shape[0] != pts.shape[0]:
        raise RuntimeError(
            f"[VIS] cluster_ids length {cluster_ids.shape[0]} != points {pts.shape[0]} "
            f"({cluster_ids_path} must align to {ply_path})"
        )

    cluster_map_json = _load_json(cluster_map_path)
    point_labels = _build_point_label_array(cluster_ids, cluster_map_json)

    labels_in_primitives = sorted({
        _norm_label(b_by_cid[cid].get("label", "unknown"))
        for cid in common_cids
        if _norm_label(b_by_cid[cid].get("label", "unknown")) != "unknown"
    })
    if not labels_in_primitives:
        labels_in_primitives = sorted(set(point_labels) - {"unknown"})

    before_color = np.array([0.2, 0.6, 1.0], dtype=np.float64)   # blue
    after_color  = np.array([1.0, 0.6, 0.2], dtype=np.float64)   # orange

    any_change_global = False

    for lab in labels_in_primitives:
        print(f"\n[VIS] Label: {lab}")

        mask = (point_labels == lab)
        if not np.any(mask):
            print("[VIS]  (no points for this label; skipping)")
            continue

        out_cols = np.zeros((pts.shape[0], 3), dtype=np.float64)  # others black
        if cols is not None and cols.shape[0] == pts.shape[0]:
            out_cols[mask] = cols[mask]
        else:
            out_cols[mask] = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        pcd_lab = o3d.geometry.PointCloud()
        pcd_lab.points = o3d.utility.Vector3dVector(pts)
        pcd_lab.colors = o3d.utility.Vector3dVector(out_cols)

        geoms = [pcd_lab]
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))

        # add only boxes for this label, and PRINT their params
        count_boxes = 0
        any_change_label = False

        for cid in common_cids:
            pb = b_by_cid[cid]
            pa = a_by_cid[cid]

            lb = _norm_label(pb.get("label", "unknown"))
            if lb != lab:
                continue

            cb, eb, Rb = _get_params(pb)
            ca, ea, Ra = _get_params(pa)

            # compute delta
            de = ea - eb
            de_norm = float(np.linalg.norm(de))

            changed = de_norm > 1e-9
            any_change_label = any_change_label or changed
            any_change_global = any_change_global or changed

            print(f"  - cluster_id={cid} label={pb.get('label','unknown')}")
            print(f"      center:  {_format_vec(cb)}")
            print(f"      extent (before): {_format_vec(eb)}")
            print(f"      extent (after) : {_format_vec(ea)}")
            print(f"      extent delta   : {_format_vec(de)}  |delta|={de_norm:.6e}")

            obb_b = o3d.geometry.OrientedBoundingBox(cb, Rb, eb)
            obb_b.color = before_color
            geoms.append(obb_b)

            # display-only nudge for AFTER so it’s visible even if identical
            # nudge along the box's x-axis direction (world): R[:,0]
            axis_x = Ra[:, 0] if Ra.shape == (3, 3) else np.array([1.0, 0.0, 0.0], dtype=np.float64)
            diag = float(np.linalg.norm(ea))
            nudge = max(display_min_nudge, display_nudge_frac * diag)
            ca_disp = ca + nudge * axis_x

            obb_a = o3d.geometry.OrientedBoundingBox(ca_disp, Ra, ea)
            obb_a.color = after_color
            geoms.append(obb_a)

            count_boxes += 1

        if count_boxes == 0:
            print("[VIS]  (no boxes for this label; skipping)")
            continue

        if not any_change_label:
            print("[VIS]  NOTE: No extent changes detected for this label (boxes may overlap perfectly).")
            print("[VIS]  (I still nudged AFTER boxes slightly for visibility.)")

        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"Label: {lab} | Before (blue) vs After (orange, nudged)",
        )

    if not any_change_global:
        print("\n[VIS] WARNING: No extent changes detected for ANY label.")
        print("[VIS] Most likely cause: DSL type names do not match primitive labels,")
        print("[VIS] so no equivalence group was applied. Check optimized_report.json.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--ply", required=True)
    args = ap.parse_args()

    run_before_after_vis_per_label(
        before_primitives_json=args.before,
        after_primitives_json=args.after,
        ply_path=args.ply,
    )
