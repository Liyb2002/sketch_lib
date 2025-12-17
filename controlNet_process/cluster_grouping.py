import numpy as np
import open3d as o3d
import os
import glob
import json
import matplotlib.pyplot as plt

# --- CONFIG ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.join(ROOT_DIR, "sketch", "3d_reconstruction")
OVERLAYS_DIR = os.path.join(ROOT_DIR, "sketch", "final_overlays")

# Inputs
FUSED_PLY_PATH = os.path.join(SCENE_DIR, "fused_model.ply")
CLUSTERS_PATH = os.path.join(SCENE_DIR, "clusters_k20.npy")

# Outputs
OUTPUT_DIR = os.path.join(SCENE_DIR, "final_segmentation")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

FINAL_PLY_PATH = os.path.join(OUTPUT_DIR, "semantic_fused_model.ply")
FINAL_JSON_PATH = os.path.join(OUTPUT_DIR, "segmentation_registry.json")

# Logic Thresholds
MERGE_THRESHOLD = 0.90  # If >90% of a cluster is Label A, force the whole cluster to Label A
IGNORE_THRESHOLD = 0.10 # If <10% of a cluster is Label B, ignore those points (set to unknown)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def match_color_to_semantic(point_color, json_data):
    """ Matches a point's color to the semantic definition in the view's JSON. """
    if np.sum(point_color) < 0.05: return None

    best_name = None
    min_dist = 0.1
    for name, data in json_data.items():
        ref_rgb = np.array(data["color_rgb_norm"])
        dist = np.linalg.norm(point_color - ref_rgb)
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name

def generate_palette(unique_labels):
    """ Generates a stable color palette for semantic parts. """
    palette = {}
    # Use tab20 for high distinction
    cmap = plt.get_cmap("tab20")
    
    sorted_labels = sorted(list(unique_labels))
    for i, label in enumerate(sorted_labels):
        if label == "unknown":
            palette[label] = [0.2, 0.2, 0.2]
        else:
            palette[label] = cmap(i % 20)[:3] # RGB
    return palette

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print(f"[INFO] Loading geometry: {FUSED_PLY_PATH}")
    pcd_orig = o3d.io.read_point_cloud(FUSED_PLY_PATH)
    points = np.asarray(pcd_orig.points)
    num_points = points.shape[0]

    print(f"[INFO] Loading original clusters: {CLUSTERS_PATH}")
    if not os.path.exists(CLUSTERS_PATH):
        print("[ERROR] Cluster file missing.")
        return
    orig_cluster_ids = np.load(CLUSTERS_PATH).reshape(-1)

    # ---------------------------------------------------------
    # 1. AGGREGATE RAW VOTES (From 6 Views)
    # ---------------------------------------------------------
    print("[INFO] Computing raw semantic labels from views...")
    
    votes_temp = [{} for _ in range(num_points)]
    global_semantic_registry = set()
    global_semantic_registry.add("unknown")

    ply_files = glob.glob(os.path.join(OVERLAYS_DIR, "*_labeled.ply"))
    
    for ply_path in ply_files:
        base_name = os.path.basename(ply_path).replace("_labeled.ply", "")
        json_path = os.path.join(OVERLAYS_DIR, f"{base_name}_labels.json")
        
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f: view_metadata = json.load(f)

        pcd_view = o3d.io.read_point_cloud(ply_path)
        colors_view = np.asarray(pcd_view.colors)
        
        # Optimize loop: only check colored points
        valid_indices = np.where(np.sum(colors_view, axis=1) > 0.05)[0]
        
        for idx in valid_indices:
            label = match_color_to_semantic(colors_view[idx], view_metadata)
            if label:
                global_semantic_registry.add(label)
                if label not in votes_temp[idx]:
                    votes_temp[idx][label] = 0
                votes_temp[idx][label] += 1

    # Resolve Majority Vote per Point
    raw_point_labels = ["unknown"] * num_points
    for i in range(num_points):
        if votes_temp[i]:
            raw_point_labels[i] = max(votes_temp[i], key=votes_temp[i].get)

    # ---------------------------------------------------------
    # 2. APPLY REFINEMENT LOGIC (Clean individual clusters)
    # ---------------------------------------------------------
    print(f"[INFO] Refining clusters (> {MERGE_THRESHOLD:.0%} dominance rule)...")
    
    refined_labels = list(raw_point_labels)
    unique_orig_clusters = np.unique(orig_cluster_ids)
    
    for cid in unique_orig_clusters:
        if cid < 0: continue 
        
        # Identify points in this cluster
        indices = np.where(orig_cluster_ids == cid)[0]
        total = len(indices)
        
        # Count labels within this cluster
        current_labels = [raw_point_labels[i] for i in indices]
        counts = {}
        for lbl in current_labels:
            if lbl != "unknown":
                counts[lbl] = counts.get(lbl, 0) + 1
        
        if not counts: continue # All unknown
        
        dom_label = max(counts, key=counts.get)
        dom_ratio = counts[dom_label] / total
        
        if dom_ratio > MERGE_THRESHOLD:
            # FORCE MERGE: Cluster is pure enough, overwrite noise
            for idx in indices:
                refined_labels[idx] = dom_label
        else:
            # SPLIT LOGIC: Keep split, but remove tiny noise
            for lbl, count in counts.items():
                ratio = count / total
                if ratio < IGNORE_THRESHOLD:
                    # Treat as noise -> set to unknown
                    subset_indices = [i for i in indices if raw_point_labels[i] == lbl]
                    for idx in subset_indices:
                        refined_labels[idx] = "unknown"

    # ---------------------------------------------------------
    # 3. RE-INDEXING (Create Final Separated Clusters)
    # ---------------------------------------------------------
    print("[INFO] Creating final registry (Separating splits)...")
    
    # We will create a new ID for every (OriginalCluster + Label) pair.
    # This keeps Cluster 1 "Leg" distinct from Cluster 2 "Leg".
    
    final_cluster_ids = np.full(num_points, -1, dtype=np.int32)
    registry = {}
    new_id_counter = 0
    
    palette = generate_palette(global_semantic_registry)

    for cid in unique_orig_clusters:
        if cid < 0: continue
        
        indices = np.where(orig_cluster_ids == cid)[0]
        
        # Identify unique labels surviving in this cluster
        unique_labels_in_cluster = set([refined_labels[i] for i in indices])
        
        for lbl in unique_labels_in_cluster:
            # Filter points belonging to this specific (Cluster, Label) pair
            mask_indices = [i for i in indices if refined_labels[i] == lbl]
            
            if not mask_indices: continue
            
            # Assign New Unique ID
            final_cluster_ids[mask_indices] = new_id_counter
            
            # Get Color
            color_rgb = palette.get(lbl, [0.2, 0.2, 0.2])
            
            # Register Metadata
            registry[new_id_counter] = {
                "label": lbl,
                "original_cluster_id": int(cid),
                "point_count": len(mask_indices),
                "color_rgb": [float(c) for c in color_rgb]
            }
            
            new_id_counter += 1

    # ---------------------------------------------------------
    # 4. EXPORT
    # ---------------------------------------------------------
    
    # Save JSON
    with open(FINAL_JSON_PATH, 'w') as f:
        json.dump(registry, f, indent=4)
    print(f"[SUCCESS] Registry saved: {FINAL_JSON_PATH}")

    # Save PLY
    final_colors = np.zeros((num_points, 3))
    
    for i in range(num_points):
        new_id = final_cluster_ids[i]
        if new_id != -1:
            lbl = registry[new_id]["label"]
            final_colors[i] = palette[lbl]
        else:
            final_colors[i] = [0.1, 0.1, 0.1] # Background/Noise
            
    pcd_orig.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.io.write_point_cloud(FINAL_PLY_PATH, pcd_orig)
    print(f"[SUCCESS] Model saved: {FINAL_PLY_PATH}")

if __name__ == "__main__":
    main()