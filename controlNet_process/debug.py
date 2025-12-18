import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_clusters(mesh_path, labels_path):
    # 1. Load the mesh
    if not Path(mesh_path).exists():
        print(f"Error: Mesh file not found at {mesh_path}")
        return
    
    mesh = o3d.io.read_point_cloud(str(mesh_path)) # Loading as PointCloud for label mapping
    # If it's a mesh and you need triangles, use o3d.io.read_triangle_mesh
    
    # 2. Load the labels
    if not Path(labels_path).exists():
        print(f"Error: Labels file not found at {labels_path}")
        return
    
    labels = np.load(labels_path)
    
    # Ensure labels match point count
    num_points = len(mesh.points)
    if len(labels) != num_points:
        print(f"Warning: Label mismatch. Points: {num_points}, Labels: {len(labels)}")
        # If labels are shorter, they might be for mesh vertices; if longer, check preprocessing.
        labels = labels[:num_points]

    # 3. Create a color map
    max_label = labels.max()
    # Using 'tab20' colormap since your K=20
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    
    # Open3D expects RGB in [0, 1] without the Alpha channel
    mesh.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 4. Render
    print(f"Visualizing {len(np.unique(labels))} clusters...")
    o3d.visualization.draw_geometries([mesh], 
                                      window_name="PartField Clustering Results",
                                      width=1280, height=720)

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    MESH_FILE = ROOT / "sketch" / "3d_reconstruction" / "fused_model.ply"
    LABEL_FILE = ROOT / "sketch" / "3d_reconstruction" / "clusters_k20.npy"
    
    visualize_clusters(MESH_FILE, LABEL_FILE)