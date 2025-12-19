#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import json
import os


def build_registry_from_label_map(label_map_path, out_registry_path):
    """
    Converts sketch/final_overlays/label_color_map.json into a registry:

    registry = {
      "0": {"label": "seat", "color_rgb": [r_norm,g_norm,b_norm]},
      "1": {"label": "backrest", "color_rgb": [...]},
      ...
    }
    """
    with open(label_map_path, "r") as f:
        lm = json.load(f)

    labels = lm.get("labels_in_order", [])
    label_to_id = lm.get("label_to_id", {})
    label_to_color_rgb_255 = lm.get("label_to_color_rgb", {})  # [R,G,B] ints 0..255

    registry = {}
    for lab in labels:
        pid = label_to_id.get(lab, None)
        if pid is None:
            continue
        rgb255 = label_to_color_rgb_255.get(lab, None)
        if rgb255 is None:
            continue
        r, g, b = rgb255
        registry[str(pid)] = {
            "label": lab,
            "color_rgb": [float(r)/255.0, float(g)/255.0, float(b)/255.0],
        }

    os.makedirs(os.path.dirname(out_registry_path), exist_ok=True)
    with open(out_registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"[REGISTRY] Saved: {out_registry_path} ({len(registry)} entries)")


def run_pca_analysis(ply_path, registry_path):
    """
    Fits PCA OrientedBoundingBoxes (Open3D) to semantic parts.
    Points are selected by matching the exact part color in the merged ply.
    """
    print(f"[PCA] Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if points.shape[0] == 0:
        print("[PCA] Empty point cloud.")
        return []

    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Registry not found at {registry_path}")

    with open(registry_path, "r") as f:
        registry = json.load(f)

    parts_db = []

    for part_id_str, info in registry.items():
        label = info.get("label", "unknown")
        if label == "unknown":
            continue

        target_color = np.array(info["color_rgb"], dtype=np.float64)

        # tolerance for float compare (colors are 0..1 floats)
        mask = np.all(np.abs(colors - target_color) < 0.005, axis=1)
        cluster_points = points[mask]

        if len(cluster_points) < 10:
            continue

        temp = o3d.geometry.PointCloud()
        temp.points = o3d.utility.Vector3dVector(cluster_points)

        obb = temp.get_oriented_bounding_box()

        parts_db.append({
            "part_id": int(part_id_str),
            "label": label,
            "parameters": {
                "center": obb.center.tolist(),
                "extent": obb.extent.tolist(),
                "rotation": obb.R.tolist(),
            },
            "color_rgb": info["color_rgb"],   # normalized, mainly for reference
            "point_count": int(len(cluster_points)),
        })

    print(f"[PCA] Processed {len(parts_db)} semantic primitives.")
    return parts_db


def save_primitives_to_json(parts_db, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(parts_db, f, indent=2)
    print(f"[PCA] Saved primitives to: {output_path}")
