import open3d as o3d
import os

def visualize_heatmap(heatmap_ply_path: str):
    """
    Visualizes a heatmap .ply file with Open3D.
    
    Args:
        heatmap_ply_path (str): The path to the .ply file containing the heatmap.
    """
    if not os.path.exists(heatmap_ply_path):
        print(f"[ERROR] The heatmap PLY file does not exist: {heatmap_ply_path}")
        return
    
    # Load the PLY file using Open3D
    pcd = o3d.io.read_point_cloud(heatmap_ply_path)
    
    # Check if the point cloud is empty
    if len(pcd.points) == 0:
        print(f"[ERROR] No points found in the heatmap PLY file: {heatmap_ply_path}")
        return

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Heatmap Visualization", width=800, height=600)
    
    # Add the point cloud to the visualization
    vis.add_geometry(pcd)
    
    # Optionally, add a coordinate frame to better understand the orientation
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(coordinate_frame)
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()
