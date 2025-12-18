import open3d as o3d
import json
import numpy as np
import os

def run_visualization(json_path):
    """
    Visualizes the generated Bounding Boxes. 
    Assumes the PLY is in the same relative structure or referenced in data.
    """
    if not os.path.exists(json_path):
        print(f"[VIS] Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        primitives = json.load(f)

    # We need the original PLY for context - going up from dsl_extraction 
    # back to final_segmentation to find the model
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(json_path)))
    ply_path = os.path.join(root_dir, "3d_reconstruction", "final_segmentation", "semantic_fused_model.ply")
    
    geometries = []
    if os.path.exists(ply_path):
        pcd = o3d.io.read_point_cloud(ply_path)
        geometries.append(pcd)
    else:
        print(f"[VIS] Warning: Context PLY not found at {ply_path}")

    for prim in primitives:
        # Reconstruct the OBB from JSON data
        obb = o3d.geometry.OrientedBoundingBox()
        obb.center = np.array(prim["center"])
        obb.extent = np.array(prim["extent"])
        obb.R = np.array(prim["rotation"])
        obb.color = np.array(prim["color_rgb"]) # Match box color to part color
        
        geometries.append(obb)

    print(f"[VIS] Showing {len(geometries)-1} boxes. Close the window to continue.")
    o3d.visualization.draw_geometries(geometries, 
                                      window_name="PCA Bounding Box Verification",
                                      width=1280, height=720)