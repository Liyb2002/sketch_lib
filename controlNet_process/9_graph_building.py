#!/usr/bin/env python3
import os
import json
import numpy as np
import open3d as o3d

from graph_building.object_space import compute_object_space, save_object_space
from graph_building.pca_analysis import run as run_pca
from graph_building.find_symmetry import find_symmetry
from graph_building.find_attachment import find_attachments
from graph_building.vis import verify_relations_vis

ROOT = os.path.dirname(os.path.abspath(__file__))

SAVE_DIR = os.path.join(ROOT, "sketch", "partfield_overlay", "label_assignment_k20")
PLY_PATH = os.path.join(SAVE_DIR, "assignment_colored.ply")
SEM_JSON = os.path.join(SAVE_DIR, "labels_semantic.json")
IDS_PATH = os.path.join(SAVE_DIR, "assigned_label_ids.npy")

OBJSPACE_JSON = os.path.join(SAVE_DIR, "object_space.json")
BBOX_GRAPH_JSON = os.path.join(SAVE_DIR, "bbox_graph.json")

# toggle verification visualization
VIS_VERIFY = True

# attachment threshold (object-space units)
ATTACH_THRESH = 0.01


def main():
    # ---- load points + assignment ----
    pcd = o3d.io.read_point_cloud(PLY_PATH)
    pts = np.asarray(pcd.points, dtype=np.float64)
    assigned_ids = np.load(IDS_PATH).reshape(-1).astype(np.int32)

    # ---- 1) object space ----
    obj_space = compute_object_space(pts)
    save_object_space(OBJSPACE_JSON, obj_space)
    print("[OBJ] saved:", OBJSPACE_JSON)

    # ---- 2) bboxes ----
    bbox_json_path = run_pca(SAVE_DIR, obj_space)
    with open(bbox_json_path, "r") as f:
        bboxes = json.load(f)

    # ---- 3) relations ----
    symmetry = find_symmetry(sorted(bboxes.keys()), ignore_unknown=True)
    attachments = find_attachments(
        bboxes_by_name=bboxes,
        object_space=obj_space,
        attach_thresh=ATTACH_THRESH,
        ignore_unknown=False,
    )

    graph = {
        "params": {"attach_thresh": float(ATTACH_THRESH)},
        "nodes": [
            {"name": n, "label_id": int(bboxes[n]["label_id"]), "n_points": int(bboxes[n]["n_points"])}
            for n in sorted(bboxes.keys())
        ],
        "symmetry": symmetry,
        "attachments": attachments,
        "bboxes_path": os.path.basename(bbox_json_path),
        "object_space_path": os.path.basename(OBJSPACE_JSON),
    }

    with open(BBOX_GRAPH_JSON, "w") as f:
        json.dump(graph, f, indent=2)

    print("[GRAPH] saved:", BBOX_GRAPH_JSON)
    print(f"[GRAPH] symmetry pairs: {len(symmetry['pairs'])}, attachments: {len(attachments)}")

    # ---- 4) verification vis ----
    if VIS_VERIFY:
        verify_relations_vis(
            pts=pts,
            assigned_ids=assigned_ids,
            bboxes_by_name=bboxes,
            symmetry=symmetry,
            attachments=attachments,
            vis_anchor_points=True,
            anchor_radius=0.002,
        )


if __name__ == "__main__":
    main()
