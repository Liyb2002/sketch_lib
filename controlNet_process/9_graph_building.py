#!/usr/bin/env python3
import os
import json
import numpy as np
import open3d as o3d

from graph_building.object_space import compute_object_space, save_object_space
from graph_building.pca_analysis import run as run_pca
from graph_building.find_symmetry import find_symmetry
from graph_building.find_attachment import find_attachments
from graph_building.find_containment import find_containment
from graph_building.save_relations import save_initial_constraints

from graph_building.attachment_face import annotate_attachment_faces, print_attachment_faces
from graph_building.vis import verify_relations_vis
from graph_building.vis_bbx import vis_bboxes_by_label

ROOT = os.path.dirname(os.path.abspath(__file__))

SAVE_DIR = os.path.join(ROOT, "sketch", "partfield_overlay", "label_assignment_k20")
PLY_PATH = os.path.join(SAVE_DIR, "assignment_colored.ply")
IDS_PATH = os.path.join(SAVE_DIR, "assigned_label_ids.npy")

OBJSPACE_JSON = os.path.join(SAVE_DIR, "object_space.json")
AEP_DIR = os.path.join(ROOT, "sketch", "AEP")

VIS_VERIFY = True

ATTACH_THRESH = 0.01

# containment params (object-space units)
CONTAIN_TOL = 1e-6
CONTAIN_STRICT = True
CONTAIN_STRICT_MARGIN = 1e-4


def main():
    pcd = o3d.io.read_point_cloud(PLY_PATH)
    pts = np.asarray(pcd.points, dtype=np.float64)
    assigned_ids = np.load(IDS_PATH).reshape(-1).astype(np.int32)

    # 1) object space
    obj_space = compute_object_space(pts)
    save_object_space(OBJSPACE_JSON, obj_space)
    print("[OBJ] saved:", OBJSPACE_JSON)

    # 2) bboxes
    bbox_json_path = run_pca(SAVE_DIR, obj_space)
    with open(bbox_json_path, "r") as f:w
        bboxes = json.load(f)


    # vis the bbx
    if VIS_VERIFY:
        pcd0 = o3d.io.read_point_cloud(PLY_PATH)
        cols0 = np.asarray(pcd0.colors, dtype=np.float64) if pcd0.has_colors() else None

        vis_bboxes_by_label(
            pts=pts,
            assigned_ids=assigned_ids,
            bboxes_by_name=bboxes,
            colors=cols0,
            window_prefix="BBX_VERIFY",
            ignore_unknown=False,
            show_all_bboxes=True,
        )


    # 3) relations
    symmetry = find_symmetry(sorted(bboxes.keys()), ignore_unknown=True)

    attachments = find_attachments(
        bboxes_by_name=bboxes,
        object_space=obj_space,
        attach_thresh=float(ATTACH_THRESH),
        ignore_unknown=False,
    )

    # 3.5) face-level attachment annotation (u0/u1/u2 in object frame)
    attachments = annotate_attachment_faces(
        bboxes_by_name=bboxes,
        attachments=attachments,
        object_space=obj_space,
        gap_tol=float(ATTACH_THRESH),
        overlap_tol=float(ATTACH_THRESH),
    )
    print("[REL] attachments annotated with faces (u0/u1/u2). Printing:")
    print_attachment_faces(attachments)

    # 3.6) containment
    containment = find_containment(
        bboxes_by_name=bboxes,
        object_space=obj_space,
        contain_tol=float(CONTAIN_TOL),
        ignore_unknown=False,
        require_strict=bool(CONTAIN_STRICT),
        strict_margin=float(CONTAIN_STRICT_MARGIN),
    )

    print(
        f"[REL] symmetry pairs: {len(symmetry.get('pairs', []))}, "
        f"attachments: {len(attachments)}, containment: {len(containment)}"
    )

    # 4) save to AEP
    out_constraints = save_initial_constraints(
        aep_dir=AEP_DIR,
        symmetry=symmetry,
        attachments=attachments,  # now includes a_face/b_face/axis/etc.
        object_space=obj_space,
        bboxes_by_name=bboxes,
        params={
            "attach_thresh": float(ATTACH_THRESH),
            "attach_face_gap_tol": float(ATTACH_THRESH),
            "attach_face_overlap_tol": float(ATTACH_THRESH),
            "contain_tol": float(CONTAIN_TOL),
            "contain_strict": bool(CONTAIN_STRICT),
            "contain_strict_margin": float(CONTAIN_STRICT_MARGIN),
        },
    )
    print("[AEP] saved:", out_constraints)

    # Append containment into saved JSON (minimal change without editing save_relations.py)
    with open(out_constraints, "r") as f:
        data = json.load(f)
    data["containment"] = containment
    tmp = out_constraints + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, out_constraints)
    print("[AEP] updated with containment:", out_constraints)

    # 5) verification vis
    # if VIS_VERIFY:
    
    #     verify_relations_vis(
    #         pts=pts,
    #         assigned_ids=assigned_ids,
    #         bboxes_by_name=bboxes,
    #         symmetry=symmetry,
    #         attachments=attachments,   # annotated with a_face/b_face = +u0 ...
    #         object_space=obj_space,
    #         containment=containment,
    #         vis_anchor_points=True,
    #         anchor_radius=0.002,
    #         ignore_unknown=False,
    #     )


if __name__ == "__main__":
    main()
