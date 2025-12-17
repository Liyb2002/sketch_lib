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
MERGE_THRESHOLD = 0.90  # >90% Dominance -> Merge whole cluster to this label
IGNORE_THRESHOLD = 0.10 # <10% Presence  -> Ignore points (set to unknown)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def match_color_to_semantic(point_color, json_data):
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
    palette = {}
    cmap = plt.get_cmap("tab20")
    sorted_labels = sorted(list(unique_labels))
    for i, label in enumerate(sorted_labels):
        if label == "unknown":
            palette[label] = [0.2, 0.2, 0.2]
        else:
            palette[label] = cmap(i % 20)[:3]
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
    print("[INFO] Computing raw semantic labels...")
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
        valid_indices = np.where(np.sum(colors_view, axis=1) > 0.05)[0]
        
        for idx in valid_indices:
            label = match_color_to_semantic(colors_view[idx], view_metadata)
            if label:
                global_semantic_registry.add(label)
                if label not in votes_temp[idx]: votes_temp[idx][label] = 0
                votes_temp[idx][label] += 1

    raw_point_labels = ["unknown"] * num_points
    for i in range(num_points):
        if votes_temp[i]:
            raw_point_labels[i] = max(votes_temp[i], key=votes_temp[i].get)

    # ---------------------------------------------------------
    # 2. APPLY REFINEMENT (Merge/Split/Ignore)
    # ---------------------------------------------------------
    print(f"[INFO] Refining clusters...")
    refined_labels = list(raw_point_labels)
    unique_orig_clusters = np.unique(orig_cluster_ids)
    
    for cid in unique_orig_clusters:
        if cid < 0: continue 
        indices = np.where(orig_cluster_ids == cid)[0]
        total = len(indices)
        current_labels = [raw_point_labels[i] for i in indices]
        
        counts = {}
        for lbl in current_labels:
            if lbl != "unknown": counts[lbl] = counts.get(lbl, 0) + 1
        if not counts: continue 
        
        dom_label = max(counts, key=counts.get)
        # If dominant label covers > 90% of the cluster, take it all
        if (counts[dom_label] / total) > MERGE_THRESHOLD:
            for idx in indices: refined_labels[idx] = dom_label
        else:
            # Otherwise, check small bits to ignore
            for lbl, count in counts.items():
                if (count / total) < IGNORE_THRESHOLD:
                    subset_indices = [i for i in indices if raw_point_labels[i] == lbl]
                    for idx in subset_indices: refined_labels[idx] = "unknown"

    # ---------------------------------------------------------
    # 3. GLOBAL SEMANTIC MERGE
    # ---------------------------------------------------------
    print("[INFO] Performing Global Semantic Merge...")
    
    final_cluster_ids = np.full(num_points, -1, dtype=np.int32)
    final_registry = {}
    current_id = 0
    palette = generate_palette(global_semantic_registry)

    # Convert list to array for fast masking
    refined_labels_arr = np.array(refined_labels)

    # A. MERGE LABELED PARTS
    # All points labeled "Leg" become one cluster, regardless of position
    unique_semantic_labels = set(refined_labels)
    if "unknown" in unique_semantic_labels:
        unique_semantic_labels.remove("unknown")
    
    for label_name in sorted(list(unique_semantic_labels)):
        mask = (refined_labels_arr == label_name)
        count = np.sum(mask)
        
        if count > 0:
            final_cluster_ids[mask] = current_id
            
            final_registry[current_id] = {
                "label": label_name,
                "type": "semantic_part",
                "point_count": int(count),
                "color_rgb": [float(c) for c in palette.get(label_name, [0.5, 0.5, 0.5])]
            }
            current_id += 1

    # B. HANDLE UNKNOWNS (Keep them separated by Original Cluster ID)
    # We don't want to merge "floor noise" with "ceiling noise"
    unknown_mask = (refined_labels_arr == "unknown")
    
    # We iterate over the ORIGINAL clusters to keep unknown blobs distinct
    for cid in unique_orig_clusters:
        if cid < 0: continue
        
        # Points that are in this original cluster AND are currently unknown
        # This preserves the geometric separation provided by PartField
        sub_mask = (orig_cluster_ids == cid) & unknown_mask
        count = np.sum(sub_mask)
        
        if count > 0:
            final_cluster_ids[sub_mask] = current_id
            
            final_registry[current_id] = {
                "label": "unknown",
                "type": "unknown_fragment",
                "original_cluster_id": int(cid),
                "point_count": int(count),
                "color_rgb": [0.2, 0.2, 0.2] # Dark Grey
            }
            current_id += 1

    # ---------------------------------------------------------
    # 4. EXPORT
    # ---------------------------------------------------------
    
    # Save JSON
    with open(FINAL_JSON_PATH, 'w') as f:
        json.dump(final_registry, f, indent=4)
    print(f"[SUCCESS] Registry saved: {FINAL_JSON_PATH}")

    # Save PLY
    final_colors = np.zeros((num_points, 3))
    
    for i in range(num_points):
        cid = final_cluster_ids[i]
        if cid != -1:
            # Use color from registry
            info = final_registry[cid]
            final_colors[i] = info["color_rgb"]
        else:
            # Truly unassigned (should be rare)
            final_colors[i] = [0.1, 0.1, 0.1] 

    pcd_orig.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.io.write_point_cloud(FINAL_PLY_PATH, pcd_orig)
    print(f"[SUCCESS] Model saved: {FINAL_PLY_PATH}")

if __name__ == "__main__":
    main()