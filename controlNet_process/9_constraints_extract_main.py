#!/usr/bin/env python3
import os

from constraints_extraction.pca_analysis import (
    build_registry_from_cluster_map,
    run_pca_analysis_on_clusters,
    save_primitives_to_json,
)
from constraints_extraction.vis_pca_bbx import run_visualization

from constraints_extraction.merge_neighbor_clusters import merge_neighboring_clusters_same_label
from constraints_extraction.infer_relations import main as infer_relations_main


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")

FINAL_DIR = os.path.join(SKETCH_ROOT, "clusters")
DSL_DIR   = os.path.join(SKETCH_ROOT, "dsl_optimize")

# Inputs from cluster_grouping stage
CLUSTERS_PLY   = os.path.join(FINAL_DIR, "labeled_clusters.ply")
CLUSTER_MAP    = os.path.join(FINAL_DIR, "cluster_to_label.json")
CLUSTER_IDS_NPY = os.path.join(FINAL_DIR, "final_cluster_ids.npy")  # <-- required

# Outputs for optimization stage
REGISTRY_JSON   = os.path.join(DSL_DIR, "registry.json")
PRIMITIVES_JSON = os.path.join(DSL_DIR, "pca_primitives.json")


def main():
    os.makedirs(DSL_DIR, exist_ok=True)

    if not os.path.exists(CLUSTERS_PLY):
        raise FileNotFoundError(f"Missing clusters ply: {CLUSTERS_PLY}")
    if not os.path.exists(CLUSTER_MAP):
        raise FileNotFoundError(f"Missing cluster_to_label.json: {CLUSTER_MAP}")
    if not os.path.exists(CLUSTER_IDS_NPY):
        raise FileNotFoundError(
            f"Missing final_cluster_ids.npy: {CLUSTER_IDS_NPY}\n"
            "This must be saved by your cluster_grouping step (per-point final cluster id, aligned to the PLY)."
        )

    # 1) Build registry (cluster_id -> label + optional color)
    build_registry_from_cluster_map(CLUSTER_MAP, REGISTRY_JSON)

    # 2) PCA/OBB per cluster id
    parts_db = run_pca_analysis_on_clusters(
        ply_path=CLUSTERS_PLY,
        cluster_ids_path=CLUSTER_IDS_NPY,
        registry_path=REGISTRY_JSON,
        min_points=10,
    )

    # 3) Save primitives
    save_primitives_to_json(parts_db, PRIMITIVES_JSON, source_ply=CLUSTERS_PLY)

    # print("\n[MAIN] PCA extraction done. Launching visualization...")

    # 4) Visualize (interactive)
    # run_visualization(PRIMITIVES_JSON, ply_path=CLUSTERS_PLY)


    print("\n[MAIN] PCA extraction done. Building neighbor graph + merging...")

    merge_outputs = merge_neighboring_clusters_same_label(
        ply_path=CLUSTERS_PLY,
        cluster_ids_path=CLUSTER_IDS_NPY,
        registry_path=REGISTRY_JSON,
        primitives_json_path=PRIMITIVES_JSON,  # not strictly required inside merge, but kept for interface clarity
        out_dir=DSL_DIR,
        neighbor_dist_thresh=0.02,
        min_points_per_cluster=10,
    )

    print("\n[MAIN] Merge done. Launching visualization of merged primitives...")

    # visualize merged primitives (overlay on the same ply)
    # run_visualization(merge_outputs["merged_primitives_json"], ply_path=merge_outputs["merged_ply"])

    print("\n[MAIN] Visualization done. Inferring label relations from sketch...")
    infer_relations_main()

if __name__ == "__main__":
    main()
