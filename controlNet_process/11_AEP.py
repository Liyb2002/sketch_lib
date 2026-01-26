#!/usr/bin/env python3
# 11_AEP.py
#
# Launcher (step 1 of iterative refactor, but behavior remains the same as before):
# - read constraints + edit
# - compute neighbors ONCE (sym / attach / contain) using find_affected_neighbors
# - run sym/contain propagation
# - run attachments propagation
# - save changes to sketch/AEP/aep_changes.json
# - vis AFTER everything (outside any future loop scaffolding)
#
# NOTE:
# - No iterative solving yet (no neighbor->neighbors expansion).
# - verbose is ALWAYS False.

import os
import json

from AEP.sym_and_containment import apply_symmetry_and_containment
from AEP.attachment import apply_attachments
from AEP.save_json import save_aep_changes
from AEP.vis import vis_from_saved_changes
from AEP.find_affect_neighbors import find_affected_neighbors

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
    # Collect neighbors ONCE (same behavior as before)
    # ------------------------------------------------------------
    all_neighbors = find_affected_neighbors(constraints=constraints, target=target)

    # ------------------------------------------------------------
    # Apply symmetry + containment edits
    # ------------------------------------------------------------
    symcon_res = apply_symmetry_and_containment(
        constraints=constraints,
        edit=edit,
        verbose=False,
    )

    # ------------------------------------------------------------
    # Apply attachments
    # ------------------------------------------------------------
    attach_res = apply_attachments(
        constraints=constraints,
        edit=edit,
        verbose=False,
    )

    # ------------------------------------------------------------
    # SAVE: target edit + neighbor changes
    # ------------------------------------------------------------
    save_aep_changes(
        aep_dir=AEP_DATA_DIR,
        target_edit=edit,
        symcon_res=symcon_res,
        attach_res=attach_res,
        out_filename=os.path.basename(AEP_CHANGES_PATH),
        constraints=constraints,
    )

    # ------------------------------------------------------------
    # VIS: outside any future loop scaffolding
    # ------------------------------------------------------------
    if DO_VIS:
        vis_from_saved_changes(
            overlay_ply_path=OVERLAY_PLY,
            nodes=nodes,
            neighbor_names=all_neighbors,
            aep_changes_json=AEP_CHANGES_PATH,
            target=target,
            window_name=f"AEP: target+neighbors (blue) + changed (red) | target={target}",
            show_overlay=True,
        )


if __name__ == "__main__":
    main()
