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
#     - apply_attachments(..., neighbor=<that>)
#     - compute + save {counter}_face_edit_change.json for this neighbor (counter starts at 1)
#       using AEP/find_face_edit_change.py (also debug-vis)
#   (NO adding new neighbors during the loop)
# - save + vis ONLY after the queue is empty
#
# NOTE:
# - verbose is ALWAYS False.

import os
import json
from collections import deque

from AEP.sym_and_containment import apply_symmetry_and_containment
from AEP.attachment import apply_attachments
from AEP.save_json import save_aep_changes
from AEP.vis import vis_from_saved_changes
from AEP.find_affect_neighbors import find_affected_neighbors
from AEP.find_next_face_edit_change import find_next_face_edit_change_and_save_and_vis

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


def _merge_neighbor_results(dst: dict, src: dict):
    """
    Merge {"symmetry": {...}, "containment": {...}} dicts by neighbor key.
    Later entries overwrite earlier ones for the same neighbor.
    """
    if not isinstance(dst, dict) or not isinstance(src, dict):
        return
    for k in ("symmetry", "containment"):
        if k not in dst or not isinstance(dst.get(k), dict):
            dst[k] = {}
        if isinstance(src.get(k), dict):
            dst[k].update(src[k])


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
    # Collect neighbors ONCE, enqueue them
    # ------------------------------------------------------------
    all_neighbors = find_affected_neighbors(constraints=constraints, target=target)
    q = deque(all_neighbors)

    # We'll aggregate results across per-neighbor calls
    symcon_res_all = {"symmetry": {}, "containment": {}}
    attach_changed_nodes_all = {}
    attach_summary_counts = {"volume": 0, "face": 0, "point": 0, "unknown": 0}
    attach_total_edges = 0
    attach_applied_any = False

    # Counter for {x}_face_edit_change.json
    face_edit_counter = 1

    # ------------------------------------------------------------
    # Process queue (NO adding new neighbors)
    # ------------------------------------------------------------
    while q:
        nb = q.popleft()

        # Track which connection type applied to this neighbor
        connection_type = None  # Will be 'attachment', 'symmetry', 'containment', or None

        # Symmetry + containment: apply ONLY to this neighbor
        symcon_res_nb = apply_symmetry_and_containment(
            constraints=constraints,
            edit=edit,
            verbose=False,
            neighbor=nb,
        )
        _merge_neighbor_results(symcon_res_all, symcon_res_nb)

        # Check if symmetry or containment produced a change for this neighbor
        if isinstance(symcon_res_nb, dict):
            if symcon_res_nb.get("symmetry", {}).get(nb) is not None:
                connection_type = "symmetry"
            elif symcon_res_nb.get("containment", {}).get(nb) is not None:
                connection_type = "containment"

        # Attachments: apply ONLY to this neighbor
        attach_res_nb = apply_attachments(
            constraints=constraints,
            edit=edit,
            verbose=False,
            neighbor=nb,
        )

        # Attachment takes precedence if it produced a change
        if isinstance(attach_res_nb, dict):
            cn = attach_res_nb.get("changed_nodes", {})
            if isinstance(cn, dict) and cn.get(nb) is not None:
                connection_type = "attachment"

        # Accumulate attachment results (changed_nodes is keyed by neighbor name)
        if isinstance(attach_res_nb, dict):
            if attach_res_nb.get("applied", False):
                attach_applied_any = True

            cn = attach_res_nb.get("changed_nodes", {})
            if isinstance(cn, dict) and cn:
                attach_changed_nodes_all.update(cn)

            summ = attach_res_nb.get("summary", {}) or {}
            attach_total_edges += int(summ.get("total_edges", 0) or 0)
            counts = summ.get("counts", {}) or {}
            for kk in ("volume", "face", "point", "unknown"):
                if kk in counts:
                    attach_summary_counts[kk] += int(counts.get(kk, 0) or 0)

        # ------------------------------------------------------------
        # Compute + save per-neighbor face_edit_change.json (and debug-vis)
        # NOW WITH connection_type information
        # Prefer attachments result if present, else sym/contain result.
        # ------------------------------------------------------------
        before_obb = None
        after_obb = None

        # Prefer attachments (if it produced an edit)
        cn = (attach_res_nb or {}).get("changed_nodes", {}) if isinstance(attach_res_nb, dict) else {}
        rec = cn.get(nb, None)
        if isinstance(rec, dict) and isinstance(rec.get("before_obb"), dict) and isinstance(rec.get("after_obb"), dict):
            before_obb = rec["before_obb"]
            after_obb = rec["after_obb"]
        else:
            # Fallback to sym/contain
            if isinstance(symcon_res_nb, dict):
                for kk in ("symmetry", "containment"):
                    m = symcon_res_nb.get(kk, {}) or {}
                    rec2 = m.get(nb, None)
                    if isinstance(rec2, dict) and isinstance(rec2.get("before_obb"), dict) and isinstance(rec2.get("after_obb"), dict):
                        before_obb = rec2["before_obb"]
                        after_obb = rec2["after_obb"]
                        break

        if before_obb is not None and after_obb is not None:
            find_next_face_edit_change_and_save_and_vis(
                aep_dir=AEP_DATA_DIR,
                counter=face_edit_counter,
                edit=edit,
                neighbor_name=nb,
                neighbor_before_obb=before_obb,
                neighbor_after_obb=after_obb,
                connection_type=connection_type,  # <-- NEW PARAMETER
                overlay_ply_path=OVERLAY_PLY,
                do_vis=True,
            )

            face_edit_counter += 1

    # Build final attachment result in the same shape expected by save_aep_changes
    attach_res_all = {
        "target": target,
        "applied": bool(attach_applied_any),
        "changed_nodes": attach_changed_nodes_all,
        "summary": {
            "total_edges": int(attach_total_edges),
            "counts": attach_summary_counts,
        },
    }

    # ------------------------------------------------------------
    # SAVE once: target edit + aggregated neighbor changes
    # ------------------------------------------------------------
    save_aep_changes(
        aep_dir=AEP_DATA_DIR,
        target_edit=edit,
        symcon_res=symcon_res_all,
        attach_res=attach_res_all,
        out_filename=os.path.basename(AEP_CHANGES_PATH),
        constraints=constraints,
    )

    # ------------------------------------------------------------
    # VIS once (after everything)
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