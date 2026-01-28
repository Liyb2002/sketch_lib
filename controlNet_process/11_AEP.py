#!/usr/bin/env python3
# 11_AEP.py
#
# Queue-based, NON-iterative neighbor processing:
# - read constraints + edit
# - compute neighbors ONCE (sym / attach / contain) using find_affected_neighbors
# - put neighbors into a queue
# - while queue not empty:
#     - pop one neighbor
#     - apply_symmetry_and_containment(..., neighbor=<that>)
#     - IF symcon didn't work, THEN apply_attachments(..., neighbor=<that>)
#     - compute + save {counter}_face_edit_change.json for this neighbor (counter starts at 1)
#       using AEP/find_face_edit_change.py (also debug-vis)
#   (NO adding new neighbors during the loop)
# - save + vis ONLY after the queue is empty
#
# NOTE:
# - verbose is ALWAYS False.
# - Priority: symmetry/containment > attachment

import os
import json
from collections import deque

from AEP.sym_and_containment import apply_symmetry_and_containment
from AEP.attachment import apply_attachments
from AEP.save_json import save_aep_changes
from AEP.vis import vis_from_saved_changes
from AEP.find_affect_neighbors import find_affected_neighbors
from AEP.find_next_face_edit_change import find_next_face_edit_change_and_save_and_vis
from AEP.accumulate_helper import (
    merge_neighbor_results,
    accumulate_attachment_results,
    extract_obb_data,
    init_attachment_accumulator,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

AEP_DATA_DIR = os.path.join(ROOT, "sketch", "AEP")
CONSTRAINTS_PATH = os.path.join(AEP_DATA_DIR, "filtered_relations.json")
EDIT_PATH = os.path.join(AEP_DATA_DIR, "target_face_edit_change.json")
AEP_CHANGES_PATH = os.path.join(AEP_DATA_DIR, "aep_changes.json")

OVERLAY_PLY = os.path.join(
    ROOT, "sketch", "partfield_overlay", "label_assignment_k20", "assignment_colored.ply"
)

DO_VIS = True
VIS_SEPARATE = True


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def main():
    if not os.path.isfile(CONSTRAINTS_PATH):
        raise FileNotFoundError(f"Missing constraints: {CONSTRAINTS_PATH}")
    if not os.path.isfile(EDIT_PATH):
        raise FileNotFoundError(f"Missing edit file: {EDIT_PATH}")

    constraints = load_json(CONSTRAINTS_PATH)
    edit = load_json(EDIT_PATH)

    target = edit.get("target", None)
    if not target:
        raise ValueError(f"Edit file missing 'target': {EDIT_PATH}")

    nodes = constraints.get("nodes", {}) or {}

    # ------------------------------------------------------------
    # Get all component names and track edited components
    # ------------------------------------------------------------
    all_component_names = set(nodes.keys())
    edited_components = {target}
    unedited_components = all_component_names - edited_components

    # ------------------------------------------------------------
    # Collect neighbors ONCE, enqueue them
    # ------------------------------------------------------------
    all_neighbors = find_affected_neighbors(constraints=constraints, target=target)
    q = deque(all_neighbors)

    # Initialize accumulators
    symcon_res_all = {"symmetry": {}, "containment": {}}
    attach_res_all = init_attachment_accumulator(target)

    # Counter for {x}_face_edit_change.json
    face_edit_counter = 1

    # ------------------------------------------------------------
    # Process queue (NO adding new neighbors)
    # ------------------------------------------------------------
    while q:
        nb = q.popleft()
        
        # Check if this component has already been edited
        if nb in edited_components:
            continue

        # Priority 1: Try symmetry + containment FIRST
        symcon_res_nb = apply_symmetry_and_containment(
            constraints=constraints,
            edit=edit,
            verbose=False,
            neighbor=nb,
        )
        merge_neighbor_results(symcon_res_all, symcon_res_nb)

        # Check if symmetry or containment produced a change
        symcon_worked = False
        if isinstance(symcon_res_nb, dict):
            if symcon_res_nb.get("symmetry", {}).get(nb) is not None:
                symcon_worked = True
            elif symcon_res_nb.get("containment", {}).get(nb) is not None:
                symcon_worked = True

        # Priority 2: Only try attachments if symcon didn't work
        attach_res_nb = None
        if not symcon_worked:
            attach_res_nb = apply_attachments(
                constraints=constraints,
                edit=edit,
                verbose=False,
                neighbor=nb,
            )
            accumulate_attachment_results(attach_res_all, attach_res_nb, nb)

        # ------------------------------------------------------------
        # Extract OBB data and connection type (symcon > attachment)
        # ------------------------------------------------------------
        before_obb, after_obb, connection_type = extract_obb_data(
            symcon_res_nb, attach_res_nb, nb
        )

        # ------------------------------------------------------------
        # Compute + save per-neighbor face_edit_change.json (and debug-vis)
        # ------------------------------------------------------------
        if before_obb is not None and after_obb is not None:
            find_next_face_edit_change_and_save_and_vis(
                aep_dir=AEP_DATA_DIR,
                counter=face_edit_counter,
                edit=edit,
                neighbor_name=nb,
                neighbor_before_obb=before_obb,
                neighbor_after_obb=after_obb,
                connection_type=connection_type,
                overlay_ply_path=OVERLAY_PLY,
                do_vis=True,
            )
            face_edit_counter += 1
            
            # Mark this component as edited
            edited_components.add(nb)
            unedited_components.discard(nb)
        print()

    # ------------------------------------------------------------
    # SAVE once: target edit + aggregated neighbor changes
    # ------------------------------------------------------------
    # save_aep_changes(
    #     aep_dir=AEP_DATA_DIR,
    #     target_edit=edit,
    #     symcon_res=symcon_res_all,
    #     attach_res=attach_res_all,
    #     out_filename=os.path.basename(AEP_CHANGES_PATH),
    #     constraints=constraints,
    # )

    # ------------------------------------------------------------
    # VIS once (after everything)
    # ------------------------------------------------------------
    # if DO_VIS:
    #     vis_from_saved_changes(
    #         overlay_ply_path=OVERLAY_PLY,
    #         nodes=nodes,
    #         neighbor_names=all_neighbors,
    #         aep_changes_json=AEP_CHANGES_PATH,
    #         target=target,
    #         window_name=f"AEP: target+neighbors (blue) + changed (red) | target={target}",
    #         show_overlay=True,
    #     )


if __name__ == "__main__":
    main()