#!/usr/bin/env python3
# 11_AEP.py
#
# Non-iterative neighbor processing:
# - Start with initial target component and edit
# - Find neighbors and propagate edits ONCE
# - Each component is edited at most once
#
# Structure:
# 1. target_component: which component was edited (initial_target)
# 2. edit: what edit was applied (initial_edit)
# 3. neighbors: which neighbors to propagate to (from find_affected_neighbors)

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
    collect_new_propagation_pairs,
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


def process_neighbor_edit(
    constraints: dict,
    target_edit: dict,
    neighbor: str,
    symcon_res_all: dict,
    attach_res_all: dict,
    face_edit_counter: int,
):
    """
    Process a single neighbor edit: apply symmetry/containment/attachment,
    extract OBB data, and save face edit change.
    
    Returns:
        tuple: (before_obb, after_obb, connection_type, new_counter)
    """
    # Priority 1: Try symmetry + containment FIRST
    symcon_res_nb = apply_symmetry_and_containment(
        constraints=constraints,
        edit=target_edit,
        verbose=False,
        neighbor=neighbor,
    )
    merge_neighbor_results(symcon_res_all, symcon_res_nb)

    # Check if symmetry or containment produced a change
    symcon_worked = False
    if isinstance(symcon_res_nb, dict):
        if symcon_res_nb.get("symmetry", {}).get(neighbor) is not None:
            symcon_worked = True
        elif symcon_res_nb.get("containment", {}).get(neighbor) is not None:
            symcon_worked = True

    # Priority 2: Only try attachments if symcon didn't work
    attach_res_nb = None
    if not symcon_worked:
        attach_res_nb = apply_attachments(
            constraints=constraints,
            edit=target_edit,
            verbose=False,
            neighbor=neighbor,
        )
        accumulate_attachment_results(attach_res_all, attach_res_nb, neighbor)

    # Extract OBB data and connection type (symcon > attachment)
    before_obb, after_obb, connection_type = extract_obb_data(
        symcon_res_nb, attach_res_nb, neighbor
    )

    # Save per-neighbor face_edit_change.json (and debug-vis)
    if before_obb is not None and after_obb is not None:
        find_next_face_edit_change_and_save_and_vis(
            aep_dir=AEP_DATA_DIR,
            counter=face_edit_counter,
            edit=target_edit,
            neighbor_name=neighbor,
            neighbor_before_obb=before_obb,
            neighbor_after_obb=after_obb,
            connection_type=connection_type,
            overlay_ply_path=OVERLAY_PLY,
            do_vis=True,
        )
        return before_obb, after_obb, connection_type, face_edit_counter + 1
    
    return None, None, None, face_edit_counter


def main():
    if not os.path.isfile(CONSTRAINTS_PATH):
        raise FileNotFoundError(f"Missing constraints: {CONSTRAINTS_PATH}")
    if not os.path.isfile(EDIT_PATH):
        raise FileNotFoundError(f"Missing edit file: {EDIT_PATH}")

    constraints = load_json(CONSTRAINTS_PATH)
    initial_edit = load_json(EDIT_PATH)

    initial_target = initial_edit.get("target", None)
    if not initial_target:
        raise ValueError(f"Edit file missing 'target': {EDIT_PATH}")

    nodes = constraints.get("nodes", {}) or {}

    # ------------------------------------------------------------
    # Initialize: 1) target_component  2) edit  3) neighbors
    # ------------------------------------------------------------
    target_component = initial_target
    edit = initial_edit
    neighbors = find_affected_neighbors(constraints=constraints, target=target_component)
    
    # Track edited components
    all_component_names = set(nodes.keys())
    edited_components = {target_component}
    
    # Initialize accumulators
    symcon_res_all = {"symmetry": {}, "containment": {}}
    attach_res_all = init_attachment_accumulator(target_component)
    
    # Counter for {x}_face_edit_change.json
    face_edit_counter = 1

    # ------------------------------------------------------------
    # Process all neighbors ONCE (non-iterative)
    # ------------------------------------------------------------
    for nb in neighbors:
        # Skip if already edited
        if nb in edited_components:
            continue
        
        # Process this neighbor
        before_obb, after_obb, connection_type, face_edit_counter = process_neighbor_edit(
            constraints=constraints,
            target_edit=edit,
            neighbor=nb,
            symcon_res_all=symcon_res_all,
            attach_res_all=attach_res_all,
            face_edit_counter=face_edit_counter,
        )
        
        # Mark as edited if successful
        if before_obb is not None and after_obb is not None:
            edited_components.add(nb)

    # ------------------------------------------------------------
    # Collect new propagation pairs from saved face_edit_change files
    # ------------------------------------------------------------
    print("\n" + "="*60)
    print("DEBUG: Collecting new propagation pairs...")
    print("="*60)
    
    new_pairs = collect_new_propagation_pairs(
        aep_dir=AEP_DATA_DIR,
        constraints=constraints,
        find_affected_neighbors_fn=find_affected_neighbors,
    )
    
    print(f"\nFound {len(new_pairs)} new propagation pair(s):\n")
    for i, (tgt, edt, nbrs) in enumerate(new_pairs, 1):
        print(f"  Pair {i}:")
        print(f"    target_component: {tgt}")
        print(f"    neighbors: {nbrs}")
        print()
    
    print("="*60 + "\n")

    # ------------------------------------------------------------
    # SAVE once: target edit + aggregated neighbor changes
    # ------------------------------------------------------------
    # save_aep_changes(
    #     aep_dir=AEP_DATA_DIR,
    #     target_edit=initial_edit,
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
    #         neighbor_names=list(edited_components - {initial_target}),
    #         aep_changes_json=AEP_CHANGES_PATH,
    #         target=initial_target,
    #         window_name=f"AEP: target+neighbors (blue) + changed (red) | target={initial_target}",
    #         show_overlay=True,
    #     )


if __name__ == "__main__":
    main()