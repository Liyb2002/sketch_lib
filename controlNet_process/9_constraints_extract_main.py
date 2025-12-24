#!/usr/bin/env python3
import os
import json

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
CLUSTERS_PLY    = os.path.join(FINAL_DIR, "labeled_clusters.ply")
CLUSTER_MAP     = os.path.join(FINAL_DIR, "cluster_to_label.json")
CLUSTER_IDS_NPY = os.path.join(FINAL_DIR, "final_cluster_ids.npy")  # required

# Outputs for optimization stage
REGISTRY_JSON        = os.path.join(DSL_DIR, "registry.json")              # pre-merge registry
MERGED_REGISTRY_JSON = os.path.join(DSL_DIR, "merged_registry.json")       # post-merge registry
PRIMITIVES_JSON      = os.path.join(DSL_DIR, "pca_primitives.json")        # primitives (post-merge)


def _write_merged_registry(merged_cluster_to_label_json: str, out_registry_json: str) -> None:
    """
    Convert merge_outputs["merged_cluster_to_label_json"] into registry.json format:
      { "0": {"label": "wheel_0"}, "1": {"label": "deck_0"}, ... }
    """
    with open(merged_cluster_to_label_json, "r") as f:
        merged_map = json.load(f)

    reg = {}
    for k, v in merged_map.items():
        # keys are merged cluster ids as strings already
        label = v.get("label", "unknown")
        reg[str(k)] = {"label": label}
        # optional: preserve point_count if you want
        if "point_count" in v:
            reg[str(k)]["point_count"] = v["point_count"]

    os.makedirs(os.path.dirname(out_registry_json), exist_ok=True)
    with open(out_registry_json, "w") as f:
        json.dump(reg, f, indent=2)


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

    # 1) Build pre-merge registry (old cluster ids)
    build_registry_from_cluster_map(CLUSTER_MAP, REGISTRY_JSON)

    # 2) Merge first (does NOT need PCA)
    print("\n[MAIN] Merging neighboring clusters (same label)...")
    merge_outputs = merge_neighboring_clusters_same_label(
        ply_path=CLUSTERS_PLY,
        cluster_ids_path=CLUSTER_IDS_NPY,
        registry_path=REGISTRY_JSON,
        primitives_json_path=os.path.join(DSL_DIR, "_unused_pca_primitives.json"),  # unused by merge, keep path
        out_dir=DSL_DIR,
        neighbor_dist_thresh=0.02,
        min_points_per_cluster=10,
    )

    merged_ply = merge_outputs["merged_ply"]
    merged_cluster_ids_npy = merge_outputs["merged_cluster_ids_npy"]
    merged_cluster_to_label_json = merge_outputs["merged_cluster_to_label_json"]

    # 3) Build *merged* registry (NEW cluster ids after merge)
    print("\n[MAIN] Building merged registry (for merged cluster ids)...")
    _write_merged_registry(merged_cluster_to_label_json, MERGED_REGISTRY_JSON)

    # 4) PCA/OBB per merged cluster id (now registry matches the ids)
    print("\n[MAIN] Running PCA/OBB analysis on merged clusters...")
    parts_db = run_pca_analysis_on_clusters(
        ply_path=merged_ply,
        cluster_ids_path=merged_cluster_ids_npy,
        registry_path=MERGED_REGISTRY_JSON,
        min_points=10,
    )

    # 5) Save primitives
    save_primitives_to_json(parts_db, PRIMITIVES_JSON, source_ply=merged_ply)

    # 6) Visualize merged primitives
    print("\n[MAIN] Launching visualization (merged)...")
    run_visualization(
        PRIMITIVES_JSON,
        ply_path=merged_ply,
        cluster_ids_path=merged_cluster_ids_npy,
        background_keep_ratio=0.3,
        show_all_boxes_faint=False,
    )

    print("\n[MAIN] Visualization done. Inferring label relations from sketch...")
    infer_relations_main()


if __name__ == "__main__":
    main()
