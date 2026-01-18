#!/usr/bin/env python3
# 11_AEP.py
#
# Updated launcher:
# - read constraints + edit
# - compute neighbors
# - run sym/contain propagation
# - save changes to sketch/AEP/aep_changes.json
# - vis reads from saved changes (not from in-memory)

import os
import json

from AEP.sym_and_containment import apply_symmetry_and_containment
from AEP.attachment import apply_attachments
from AEP.save_json import save_aep_changes
from AEP.vis import vis_from_saved_changes

ROOT = os.path.dirname(os.path.abspath(__file__))

AEP_DATA_DIR = os.path.join(ROOT, "sketch", "AEP")
CONSTRAINTS_PATH = os.path.join(AEP_DATA_DIR, "initial_constraints.json")
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
    symmetry = constraints.get("symmetry", {}) or {}
    attachments = constraints.get("attachments", []) or []
    containment = constraints.get("containment", []) or []

    # ------------------------------------------------------------
    # Collect neighbors (sym / attach / contain)
    # ------------------------------------------------------------
    sym_neighbors = set()
    for p in symmetry.get("pairs", []) or []:
        a = p.get("a")
        b = p.get("b")
        if a == target and b:
            sym_neighbors.add(b)
        elif b == target and a:
            sym_neighbors.add(a)

    attach_neighbors = set()
    for e in attachments:
        a = e.get("a")
        b = e.get("b")
        if a == target and b:
            attach_neighbors.add(b)
        elif b == target and a:
            attach_neighbors.add(a)

    contain_neighbors = set()
    for c in containment:
        outer = c.get("outer")
        inner = c.get("inner")
        if outer == target and inner:
            contain_neighbors.add(inner)
        elif inner == target and outer:
            contain_neighbors.add(outer)

    all_neighbors = sorted(sym_neighbors | attach_neighbors | contain_neighbors)

    # ------------------------------------------------------------
    # Apply symmetry + containment edits (and print verification)
    # ------------------------------------------------------------
    symcon_res = apply_symmetry_and_containment(constraints=constraints, edit=edit, verbose=True)

    # ------------------------------------------------------------
    # Attachments: still stub-print only (per your instruction)
    # ------------------------------------------------------------
    attach_res = apply_attachments(constraints=constraints, edit=edit, verbose=True)


    # ------------------------------------------------------------
    # SAVE: target edit + neighbor changes
    # ------------------------------------------------------------
    save_aep_changes(
        aep_dir=AEP_DATA_DIR,
        target_edit=edit,
        symcon_res=symcon_res,
        attach_res=attach_res,     # NEW
        out_filename=os.path.basename(AEP_CHANGES_PATH),
        )


    # ------------------------------------------------------------
    # VIS: neighbors in blue, changed neighbors in red
    #     (reads from saved json, not from symcon_res)
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
