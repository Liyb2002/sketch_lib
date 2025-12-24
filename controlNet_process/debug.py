#!/usr/bin/env python3
import os
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Load necessary files
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.join(ROOT_DIR, "sketch", "3d_reconstruction")
PARTFIELD_OVERLAY_DIR = os.path.join(ROOT_DIR, "sketch", "partfield_overlay")
CLUSTERS_PATH = os.path.join(SCENE_DIR, "clustering_k20.npy")
FINAL_PLY_PATH = os.path.join(SCENE_DIR, "fused_model.ply")

# Load the point cloud and clusters
pcd = o3d.io.read_point_cloud(FINAL_PLY_PATH)
points = np.asarray(pcd.points)

# Assuming final_cluster_ids contains the cluster IDs for each point
# Here, for demonstration, we load the clusters from a file
final_cluster_ids = np.load(CLUSTERS_PATH).reshape(-1)

# Create a color map where 'unknown' clusters will be visualized in red
red_color = [1.0, 0.0, 0.0]  # RGB Red
output_colors = np.asarray(pcd.colors)

# Loop through all the clusters and find "unknown" clusters
unique_cluster_ids = np.unique(final_cluster_ids)

# Iterate through the clusters and visualize "unknown" clusters in red
for cid in unique_cluster_ids:
    # Identify the points belonging to this cluster
    cluster_points_indices = np.where(final_cluster_ids == cid)[0]

    # If this cluster is labeled as "unknown", color it red
    if str(cid).startswith("unknown"):
        # Set the color of these points to red
        output_colors[cluster_points_indices] = red_color

# Update the point cloud with the new colors
pcd.colors = o3d.utility.Vector3dVector(output_colors)

# Visualize the result
o3d.visualization.draw_geometries([pcd], window_name="Unknown Clusters Visualization")
