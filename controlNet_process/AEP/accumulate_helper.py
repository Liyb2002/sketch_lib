#!/usr/bin/env python3
# AEP/accumulate_helper.py
#
# Helper functions for accumulating results from neighbor processing


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