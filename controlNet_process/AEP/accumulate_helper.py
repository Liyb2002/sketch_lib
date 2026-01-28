#!/usr/bin/env python3
# AEP/accumulate_helper.py
#
# Helper functions for accumulating results from neighbor processing

import os
import json
import glob


def merge_neighbor_results(dst: dict, src: dict):
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


def accumulate_attachment_results(
    attach_res_all: dict,
    attach_res_nb: dict,
    neighbor: str,
):
    """
    Accumulate attachment results from a single neighbor into the master dict.
    
    Args:
        attach_res_all: Master dict with keys {applied, changed_nodes, summary}
        attach_res_nb: Single neighbor's attachment result
        neighbor: Neighbor name (for tracking)
    """
    if not isinstance(attach_res_nb, dict):
        return
    
    # Track if any attachment was applied
    if attach_res_nb.get("applied", False):
        attach_res_all["applied"] = True
    
    # Accumulate changed_nodes
    cn = attach_res_nb.get("changed_nodes", {})
    if isinstance(cn, dict) and cn:
        attach_res_all["changed_nodes"].update(cn)
    
    # Accumulate summary statistics
    summ = attach_res_nb.get("summary", {}) or {}
    attach_res_all["summary"]["total_edges"] += int(summ.get("total_edges", 0) or 0)
    
    counts = summ.get("counts", {}) or {}
    for kk in ("volume", "face", "point", "unknown"):
        if kk in counts:
            attach_res_all["summary"]["counts"][kk] += int(counts.get(kk, 0) or 0)


def extract_obb_data(symcon_res_nb: dict, attach_res_nb: dict, neighbor: str):
    """
    Extract before/after OBB data for a neighbor.
    Priority: symmetry/containment > attachment
    
    Args:
        symcon_res_nb: Symmetry/containment result for this neighbor
        attach_res_nb: Attachment result for this neighbor
        neighbor: Neighbor name
    
    Returns:
        tuple: (before_obb, after_obb, connection_type) or (None, None, None)
    """
    before_obb = None
    after_obb = None
    connection_type = None
    
    # First try: symmetry/containment (priority)
    if isinstance(symcon_res_nb, dict):
        for kk in ("symmetry", "containment"):
            m = symcon_res_nb.get(kk, {}) or {}
            rec = m.get(neighbor, None)
            if isinstance(rec, dict) and isinstance(rec.get("before_obb"), dict) and isinstance(rec.get("after_obb"), dict):
                before_obb = rec["before_obb"]
                after_obb = rec["after_obb"]
                connection_type = kk
                break
    
    # Fallback to attachment if symcon didn't produce results
    if before_obb is None and isinstance(attach_res_nb, dict):
        cn = attach_res_nb.get("changed_nodes", {})
        rec = cn.get(neighbor, None)
        if isinstance(rec, dict) and isinstance(rec.get("before_obb"), dict) and isinstance(rec.get("after_obb"), dict):
            before_obb = rec["before_obb"]
            after_obb = rec["after_obb"]
            connection_type = "attachment"
    
    return before_obb, after_obb, connection_type


def init_attachment_accumulator(target: str):
    """
    Initialize the attachment results accumulator.
    
    Args:
        target: Target node name
    
    Returns:
        dict: Initialized accumulator with structure matching attach_res format
    """
    return {
        "target": target,
        "applied": False,
        "changed_nodes": {},
        "summary": {
            "total_edges": 0,
            "counts": {"volume": 0, "face": 0, "point": 0, "unknown": 0},
        },
    }


def collect_new_propagation_pairs(aep_dir: str, constraints: dict, find_affected_neighbors_fn):
    """
    Read all {counter}_face_edit_change.json files and create new propagation pairs.
    
    Each file represents a neighbor that was edited. For each:
    - target_component: the neighbor that was edited (from "target" field)
    - edit: the edit applied to that neighbor (from "change" field)
    - neighbors: find new neighbors of this target_component
    
    Args:
        aep_dir: Directory containing face_edit_change files
        constraints: Constraint graph for finding neighbors
        find_affected_neighbors_fn: Function to find neighbors
    
    Returns:
        list: List of (target_component, edit, neighbors) tuples
    """
    pairs = []
    
    # Find all *_face_edit_change.json files (e.g., 1_face_edit_change.json, 2_face_edit_change.json)
    pattern = os.path.join(aep_dir, "*_face_edit_change.json")
    files = sorted(glob.glob(pattern))
    
    for filepath in files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            # Extract target_component (the neighbor that was edited)
            target_component = data.get("target")
            if not target_component:
                continue
            
            # Extract edit (the change that was applied)
            # We need to restructure it to match the format expected by propagation functions
            change = data.get("change", {})
            if not change:
                continue
            
            # Build edit dict in the format expected by apply_symmetry_and_containment, etc.
            edit = {
                "target": target_component,
                "change": change,
            }
            
            # Find neighbors of this newly edited component
            neighbors = find_affected_neighbors_fn(constraints=constraints, target=target_component)
            
            pairs.append((target_component, edit, neighbors))
            
        except Exception as e:
            print(f"Warning: Failed to read {filepath}: {e}")
            continue
    
    return pairs


def filter_propagation_pairs(pairs: list, edited_components: set):
    """
    Filter propagation pairs to only keep valid ones.
    
    Rules:
    1. Ignore pairs where target_component starts with "unknown"
    2. Filter out neighbors that have already been edited
    3. Only keep pairs that have at least one valid neighbor
    
    Args:
        pairs: List of (target_component, edit, neighbors) tuples
        edited_components: Set of components that have already been edited
    
    Returns:
        list: Filtered list of (target_component, edit, filtered_neighbors) tuples
    """
    filtered_pairs = []
    
    for target_component, edit, neighbors in pairs:
        # Rule 1: Skip if target_component is unknown
        if target_component.startswith("unknown"):
            continue
        
        # Rule 2: Filter out neighbors that have already been edited
        valid_neighbors = [nb for nb in neighbors if nb not in edited_components]
        
        # Rule 3: Only keep pairs with at least one valid neighbor
        if len(valid_neighbors) > 0:
            filtered_pairs.append((target_component, edit, valid_neighbors))
    
    return filtered_pairs