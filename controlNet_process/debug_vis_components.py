#!/usr/bin/env python3
import os
import json
from typing import Dict, Any, List

import numpy as np
import open3d as o3d


# =========================
# Code 1: visualization lib
# =========================

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _sanitize_filename(name: str) -> str:
    # Keep it simple + stable for files
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    out = name
    for b in bad:
        out = out.replace(b, "_")
    return out


def _load_merged_cluster_to_label(path: str) -> Dict[int, str]:
    """
    merged_cluster_to_label.json:
      { "0": {"label": "...", "members":[...], "point_count": ...}, ... }
    Returns {0: "label", ...}
    """
    data = _load_json(path)
    out: Dict[int, str] = {}
    if not isinstance(data, dict):
        return out
    for k, v in data.items():
        try:
            cid = int(k)
        except Exception:
            continue
        if isinstance(v, dict):
            out[cid] = str(v.get("label", "unknown"))
        else:
            out[cid] = "unknown"
    return out


def _label_to_rgb01(label: str) -> np.ndarray:
    # deterministic pseudo-color by label hashing
    # NOTE: Python's built-in hash() may vary between processes unless PYTHONHASHSEED is fixed.
    import colorsys
    h = abs(hash(label)) % 360
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.80, 1.00)
    return np.array([r, g, b], dtype=np.float64)


def _cluster_to_rgb01(cid: int) -> np.ndarray:
    import colorsys
    h = (cid * 37) % 360
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.80, 1.00)
    return np.array([r, g, b], dtype=np.float64)


def visualize_components(
    *,
    ply_path: str,
    cluster_ids_path: str,
    merged_cluster_to_label_json: str,
    out_dir: str,
    by: str = "label",            # "label" or "cluster"
    also_show_by_cluster: bool = False,
    min_points: int = 50,
) -> Dict[str, Any]:
    """
    Visualize components based on the *actual points*, not bounding boxes.

    - by="label": for each semantic label, show its points colored, all others black.
    - by="cluster": for each merged cluster id, show its points colored, all others black.

    Saves per-component PLYs to out_dir/labels/ or out_dir/clusters/.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ply_path):
        raise FileNotFoundError(ply_path)
    if not os.path.exists(cluster_ids_path):
        raise FileNotFoundError(cluster_ids_path)
    if not os.path.exists(merged_cluster_to_label_json):
        raise FileNotFoundError(merged_cluster_to_label_json)

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        raise RuntimeError("Empty point cloud")

    cluster_ids = np.load(cluster_ids_path).reshape(-1)
    if cluster_ids.shape[0] != points.shape[0]:
        raise RuntimeError(
            "cluster_ids.npy must align 1:1 with ply point order\n"
            f"  points: {points.shape[0]}\n"
            f"  cluster_ids: {cluster_ids.shape[0]}\n"
            f"  ply_path: {ply_path}\n"
            f"  cluster_ids_path: {cluster_ids_path}\n"
        )

    cid_to_label = _load_merged_cluster_to_label(merged_cluster_to_label_json)

    # Build masks
    present_cids = sorted(set(int(x) for x in np.unique(cluster_ids) if int(x) >= 0))
    cid_counts = {cid: int(np.sum(cluster_ids == cid)) for cid in present_cids}

    # label -> point indices
    label_to_indices: Dict[str, List[int]] = {}
    for cid in present_cids:
        if cid not in cid_to_label:
            continue
        if cid_counts[cid] < min_points:
            continue
        lab = cid_to_label[cid]
        idxs = np.where(cluster_ids == cid)[0].tolist()
        label_to_indices.setdefault(lab, []).extend(idxs)

    # Sort labels by descending point count
    labels_sorted = sorted(label_to_indices.keys(), key=lambda k: len(label_to_indices[k]), reverse=True)

    # Always visualize labels first
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    print(f"\n[VIS] Total points: {points.shape[0]}")
    print(f"[VIS] Labels found (min_points per cluster={min_points}): {len(labels_sorted)}")

    for i, lab in enumerate(labels_sorted):
        idxs = np.array(label_to_indices[lab], dtype=np.int64)
        if idxs.size == 0:
            continue

        print("\n" + "=" * 80)
        print(f"[VIS][LABEL {i:03d}] {lab}")
        print(f"  points: {idxs.size}")

        colors = np.zeros((points.shape[0], 3), dtype=np.float64) + 0.02  # black-ish
        colors[idxs] = _label_to_rgb01(lab)

        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(points)
        out_pcd.colors = o3d.utility.Vector3dVector(colors)

        ply_out = os.path.join(labels_dir, f"{_sanitize_filename(lab)}.ply")
        o3d.io.write_point_cloud(ply_out, out_pcd)
        print(f"  saved: {ply_out}")

        o3d.visualization.draw_geometries(
            [out_pcd],
            window_name=f"Label: {lab} (colored) | others black",
        )

    # Optional: also visualize per merged cluster id
    if also_show_by_cluster or by == "cluster":
        clusters_dir = os.path.join(out_dir, "clusters")
        os.makedirs(clusters_dir, exist_ok=True)

        eligible_cids = [cid for cid in present_cids if cid in cid_to_label and cid_counts[cid] >= min_points]
        print(f"\n[VIS] Clusters eligible (>= {min_points} pts): {len(eligible_cids)}")

        for j, cid in enumerate(eligible_cids):
            lab = cid_to_label.get(cid, "unknown")
            idxs = np.where(cluster_ids == cid)[0]
            if idxs.size == 0:
                continue

            print("\n" + "-" * 80)
            print(f"[VIS][CLUSTER {j:03d}] cid={cid} label={lab} points={idxs.size}")

            colors = np.zeros((points.shape[0], 3), dtype=np.float64) + 0.02
            colors[idxs] = _cluster_to_rgb01(cid)

            out_pcd = o3d.geometry.PointCloud()
            out_pcd.points = o3d.utility.Vector3dVector(points)
            out_pcd.colors = o3d.utility.Vector3dVector(colors)

            ply_out = os.path.join(clusters_dir, f"cid_{cid:04d}__{_sanitize_filename(lab)}.ply")
            o3d.io.write_point_cloud(ply_out, out_pcd)
            print(f"  saved: {ply_out}")

            o3d.visualization.draw_geometries(
                [out_pcd],
                window_name=f"Cluster: {cid} ({lab}) | colored points only",
            )

    return {
        "out_dir": out_dir,
        "num_labels": len(labels_sorted),
        "labels_dir": labels_dir,
    }


# =========================
# Code 2: main runner
# =========================

def _pick_inputs(sketch_root: str) -> Dict[str, str]:
    """
    Pick a consistent set of inputs.

    Prefer merged outputs ONLY if all required merged files exist together:
      - merged_labeled_clusters.ply
      - merged_cluster_ids.npy
      - merged_cluster_to_label.json (required by this visualization)

    Otherwise fall back to base:
      - labeled_clusters.ply
      - final_cluster_ids.npy
    but still require merged_cluster_to_label.json (same as your original code).
    """
    clusters_dir = os.path.join(sketch_root, "clusters")
    dsl_dir = os.path.join(sketch_root, "dsl_optimize")

    merged_ply = os.path.join(dsl_dir, "merged_labeled_clusters.ply")
    merged_ids = os.path.join(dsl_dir, "merged_cluster_ids.npy")
    merged_map = os.path.join(dsl_dir, "merged_cluster_to_label.json")

    base_ply = os.path.join(clusters_dir, "labeled_clusters.ply")
    base_ids = os.path.join(clusters_dir, "final_cluster_ids.npy")

    has_merged_set = os.path.exists(merged_ply) and os.path.exists(merged_ids) and os.path.exists(merged_map)

    if has_merged_set:
        ply_path = merged_ply
        ids_path = merged_ids
        map_path = merged_map
        out_dir = os.path.join(dsl_dir, "component_vis")
        mode = "merged"
    else:
        ply_path = base_ply
        ids_path = base_ids
        map_path = merged_map  # REQUIRED (kept same behavior as your code 2)
        out_dir = os.path.join(dsl_dir, "component_vis")
        mode = "base"

    return {
        "mode": mode,
        "clusters_dir": clusters_dir,
        "dsl_dir": dsl_dir,
        "ply_path": ply_path,
        "cluster_ids_npy": ids_path,
        "label_map_json": map_path,
        "out_dir": out_dir,
    }


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sketch_root = os.path.join(this_dir, "sketch")

    cfg = _pick_inputs(sketch_root)

    print("\n[MAIN] === Visualize each component (by semantic label) ===")
    print("[MAIN] mode:", cfg["mode"])
    print("[MAIN] ply:", cfg["ply_path"])
    print("[MAIN] cluster_ids:", cfg["cluster_ids_npy"])
    print("[MAIN] merged_cluster_to_label:", cfg["label_map_json"])
    print("[MAIN] out_dir:", cfg["out_dir"])

    if not os.path.exists(cfg["ply_path"]):
        raise FileNotFoundError(f"Missing PLY: {cfg['ply_path']}")
    if not os.path.exists(cfg["cluster_ids_npy"]):
        raise FileNotFoundError(f"Missing cluster ids: {cfg['cluster_ids_npy']}")
    if not os.path.exists(cfg["label_map_json"]):
        raise FileNotFoundError(
            f"Missing merged_cluster_to_label.json: {cfg['label_map_json']}\n"
            "This step expects merged outputs."
        )

    visualize_components(
        ply_path=cfg["ply_path"],
        cluster_ids_path=cfg["cluster_ids_npy"],
        merged_cluster_to_label_json=cfg["label_map_json"],
        out_dir=cfg["out_dir"],
        by="label",                 # "label" is what you asked
        also_show_by_cluster=False, # set True if you want per-cluster too
        min_points=50,
    )

    print("\n[MAIN] Done.")


if __name__ == "__main__":
    main()
