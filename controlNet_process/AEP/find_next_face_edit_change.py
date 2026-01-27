#!/usr/bin/env python3
# AEP/find_next_face_edit_change.py
#
# Main launcher for finding next face edit changes.
# Dispatches to helper modules based on connection type.

from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional

from AEP.find_next_face.find_attachment_face import find_attachment_face
from AEP.find_next_face.find_sym_and_containment_face import find_sym_and_containment_face
from AEP.find_next_face.vis_next_face import vis_next_face_edit_change


def save_face_edit_change_json(
    aep_dir: str, 
    counter: int, 
    local_counter: int,
    face_edit_change: Dict[str, Any]
) -> str:
    """Save face_edit_change to {counter}_{local_counter}_face_edit_change.json"""
    os.makedirs(aep_dir, exist_ok=True)
    out_path = os.path.join(aep_dir, f"{int(counter)}_{int(local_counter)}_face_edit_change.json")
    with open(out_path, "w") as f:
        json.dump(face_edit_change, f, indent=2)
    return out_path


def find_next_face_edit_change_and_save_and_vis(
    aep_dir: str,
    counter: int,
    edit: Dict[str, Any],
    neighbor_name: str,
    neighbor_before_obb: Dict[str, Any],
    neighbor_after_obb: Dict[str, Any],
    connection_type: Optional[str] = None,  # 'attachment', 'symmetry', 'containment', or None
    overlay_ply_path: Optional[str] = None,
    do_vis: bool = True,
) -> List[str]:
    """
    Main entry point. Dispatches based on connection_type:
      - attachment: find passive face (opposite of attachment)
      - symmetry/containment: find corresponding face (same as target's edited face)
    
    Returns list of saved file paths (one per face).
    """
    
    if connection_type == "attachment":
        fec_list = find_attachment_face(
            edit=edit,
            neighbor_before_obb=neighbor_before_obb,
            neighbor_after_obb=neighbor_after_obb,
        )
        
    elif connection_type in ("symmetry", "containment"):
        fec_list = find_sym_and_containment_face(
            edit=edit,
            neighbor_before_obb=neighbor_before_obb,
            neighbor_after_obb=neighbor_after_obb,
            connection_type=connection_type,
        )
        
    else:
        print(f"Warning: Unknown connection_type '{connection_type}' for neighbor={neighbor_name}")
        return []

    # Save each face with local counter
    out_paths = []
    for local_counter, fec in enumerate(fec_list):
        # Add target name to result
        fec["target"] = neighbor_name

        # Save
        out_path = save_face_edit_change_json(
            aep_dir=aep_dir, 
            counter=counter, 
            local_counter=local_counter,
            face_edit_change=fec
        )
        out_paths.append(out_path)

        # Visualize
        # if do_vis:
        #     vis_next_face_edit_change(
        #         fec,
        #         overlay_ply_path=overlay_ply_path,
        #         show_overlay=True,
        #         window_name=f"NextFaceEdit#{counter}_{local_counter} | neighbor={neighbor_name} | type={connection_type}",
        #     )

    return out_paths