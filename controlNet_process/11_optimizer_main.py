import os
from constraints_optimization.vis import visualize_heatmap

# Define paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")
DSL_DIR      = os.path.join(SKETCH_ROOT, "dsl_optimize")
OUT_DIR = os.path.join(DSL_DIR, "optimize_iteration", "iter_000")

# Heatmap paths
HEATMAPS_DIR = os.path.join(OUT_DIR, "heat_map", "heatmaps")

def load_and_visualize_heatmaps():
    """
    This function loads the heatmaps and visualizes them using the visualize_heatmap function.
    """
    # List the directories inside heatmaps to load all the label heatmaps
    label_dirs = [d for d in os.listdir(HEATMAPS_DIR) if os.path.isdir(os.path.join(HEATMAPS_DIR, d))]
    
    print(f"[DEBUG] Found label directories: {label_dirs}")
    
    # Loop through each label directory and visualize its heatmap
    for label in label_dirs:
        heatmap_ply = os.path.join(HEATMAPS_DIR, label, f"heat_map_{label}.ply")
        if os.path.exists(heatmap_ply):
            print(f"[VIS] Visualizing heatmap for label: {label} at {heatmap_ply}")
            # Call the visualization helper function from vis.py
            visualize_heatmap(heatmap_ply)
        else:
            print(f"[WARNING] No heatmap PLY found for label: {label}")

def main():
    print("\n[OPT] === Shrink-only no-overlap + maximize size (tolerant, nonlinear) ===")
    print("[OPT] out_dir   :", OUT_DIR)

    if not os.path.exists(OUT_DIR):
        raise FileNotFoundError(f"Missing output directory: {OUT_DIR}")
    
    # Load and visualize the heatmaps
    load_and_visualize_heatmaps()



    # outputs = apply_no_overlapping_shrink_only(
    #     primitives_json_path=PRIMITIVES_JSON,
    #     out_dir=OUT_DIR,
    #     steps=1200,
    #     lr=1e-2,
    #     overlap_tol_ratio=0.02,
    #     overlap_scale_ratio=0.01,
    #     r_min=0.70,
    #     w_overlap=1.0,
    #     w_cut=50.0,
    #     w_size=2.0,
    #     w_floor=10.0,
    #     extent_floor=1e-3,
    #     min_points=10,
    #     device="cuda",
    #     verbose_every=50,
    # )

    # print("\n[OPT] Done.")
    # print("[OPT] optimized_primitives:", outputs["optimized_primitives_json"])
    # print("[OPT] report              :", outputs["report_json"])

    # print("\n[VIS] Per-label before/after views...")
    # run_before_after_vis_per_label(
    #     before_primitives_json=PRIMITIVES_JSON,
    #     after_primitives_json=outputs["optimized_primitives_json"],
    #     ply_path=PLY_PATH,
    #     cluster_ids_path=CLUSTER_IDS_NPY,
    # )

    # print("\n[VIS] Global view: all optimized boxes together...")
    # run_global_vis_all_boxes(
    #     primitives_json=outputs["optimized_primitives_json"],
    #     ply_path=PLY_PATH,
    #     show_points=True,
    # )


if __name__ == "__main__":
    main()