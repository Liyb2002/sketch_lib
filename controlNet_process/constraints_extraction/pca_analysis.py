import numpy as np
import open3d as o3d
import json
import os

def run_pca_analysis(ply_path, registry_path):
    """
    Fits Oriented Bounding Boxes (OBB) to semantic clusters using PCA.
    """
    print(f"[PCA] Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Registry not found at {registry_path}")

    with open(registry_path, 'r') as f:
        registry = json.load(f)

    parts_db = []

    for cluster_id, info in registry.items():
        label = info["label"]
        if label == "unknown":
            continue
            
        # Extract points belonging to this cluster based on the stored color
        target_color = np.array(info["color_rgb"])
        # Use a small tolerance for float comparison
        mask = np.all(np.abs(colors - target_color) < 0.005, axis=1)
        cluster_points = points[mask]

        if len(cluster_points) < 10:  # Skip tiny fragments
            continue

        # Create a temporary PCD for Open3D's OBB calculation
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        
        # PCA-based Oriented Bounding Box
        obb = temp_pcd.get_oriented_bounding_box()
        
        parts_db.append({
            "part_id": int(cluster_id),
            "label": label,
            "center": obb.center.tolist(),
            "extent": obb.extent.tolist(),   # Length, Width, Height
            "rotation": obb.R.tolist(),     # 3x3 Rotation Matrix
            "color_rgb": info["color_rgb"],
            "point_count": len(cluster_points)
        })

    print(f"[PCA] Processed {len(parts_db)} semantic primitives.")
    return parts_db

def save_primitives_to_json(parts_db, output_path):
    with open(output_path, 'w') as f:
        json.dump(parts_db, f, indent=4)
    print(f"[PCA] Saved primitives to: {output_path}")