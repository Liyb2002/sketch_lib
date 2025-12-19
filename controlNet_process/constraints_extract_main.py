#!/usr/bin/env python3
import os

from constraints_extraction.pca_analysis import (
    build_registry_from_label_map,
    run_pca_analysis,
    save_primitives_to_json,
)
from constraints_extraction.vis_pca_bbx import run_visualization


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SKETCH_ROOT = os.path.join(THIS_DIR, "sketch")

FINAL_DIR = os.path.join(SKETCH_ROOT, "final_overlays")
DSL_DIR   = os.path.join(SKETCH_ROOT, "dsl_optimize")

MERGED_PLY = os.path.join(FINAL_DIR, "merged_labeled.ply")
LABEL_MAP  = os.path.join(FINAL_DIR, "label_color_map.json")

REGISTRY_JSON   = os.path.join(DSL_DIR, "registry.json")
PRIMITIVES_JSON = os.path.join(DSL_DIR, "pca_primitives.json")


def main():
    os.makedirs(DSL_DIR, exist_ok=True)

    if not os.path.exists(MERGED_PLY):
        raise FileNotFoundError(f"Missing merged ply: {MERGED_PLY}")
    if not os.path.exists(LABEL_MAP):
        raise FileNotFoundError(f"Missing label map json: {LABEL_MAP}")

    # 1) Build registry
    build_registry_from_label_map(LABEL_MAP, REGISTRY_JSON)

    # 2) PCA → OBB per semantic part
    parts_db = run_pca_analysis(MERGED_PLY, REGISTRY_JSON)

    # 3) Save primitives
    save_primitives_to_json(parts_db, PRIMITIVES_JSON)

    print("\n[MAIN] PCA extraction done. Launching visualization...")

    # 4) Visualize (interactive)
    run_visualization(PRIMITIVES_JSON)


if __name__ == "__main__":
    main()
