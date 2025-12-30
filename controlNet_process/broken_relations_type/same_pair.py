# broken_relations_type/same_pair.py
"""
Broken same_pair detection.

User rule:
- same_pair fails as long as the sizes of two bounding boxes exceed it.

Thresholds available in graph_building/thresholds.py only give:
- same_pair_relation_threshold_confidence (for keeping relation) and floating_point_eps.
No explicit size tolerance is defined there, so we implement the simplest:
- broken if max-abs per-axis size difference > size_eps (default = floating_point_eps)

Size is taken from graph node extents (preferred) because same_pair is "same object class size".
Fallback: size from AABB side lengths if extent missing.
"""

from typing import Any, Dict, List, Tuple
import numpy as np


def _np3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)

def _node_size_vec(node: Dict[str, Any]) -> np.ndarray:
    if "extent" in node:
        try:
            e = _np3(node.get("extent", [0, 0, 0]))
            if np.all(np.isfinite(e)):
                return e
        except Exception:
            pass

    # fallback: AABB size
    aabb = node.get("aabb_world", {}) if isinstance(node.get("aabb_world", {}), dict) else {}
    mn = _np3(aabb.get("min", [0, 0, 0]))
    mx = _np3(aabb.get("max", [0, 0, 0]))
    return np.maximum(mx - mn, 0.0)

def _label_to_node(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for n in graph.get("nodes", []):
        lab = str(n.get("label", ""))
        if lab:
            out[lab] = n
    return out


def detect_broken_same_pairs(graph: Dict[str, Any], size_eps: float) -> List[Dict[str, Any]]:
    """
    Returns list of broken same_pair relations.

    Each output record includes:
      - a_label, b_label
      - size_a, size_b
      - abs_diff
      - broken: True
    """
    label_to_node = _label_to_node(graph)

    broken: List[Dict[str, Any]] = []
    for e in graph.get("edges", []):
        if str(e.get("type", "")) != "same_pair":
            continue

        a_label = str(e.get("a_label", ""))
        b_label = str(e.get("b_label", ""))
        if not a_label or not b_label:
            continue
        if a_label not in label_to_node or b_label not in label_to_node:
            continue

        na = label_to_node[a_label]
        nb = label_to_node[b_label]
        sa = _node_size_vec(na)
        sb = _node_size_vec(nb)

        diff = np.abs(sa - sb)
        if float(np.max(diff)) > float(size_eps):
            broken.append({
                "type": "same_pair",
                "a_label": a_label,
                "b_label": b_label,
                "a_id": int(na.get("id", -1)),
                "b_id": int(nb.get("id", -1)),
                "size_a": sa.tolist(),
                "size_b": sb.tolist(),
                "abs_diff": diff.tolist(),
                "threshold": float(size_eps),
                "broken": True,
                "confidence": float(e.get("confidence", 1.0)),
                "evidence": e.get("evidence", ""),
            })

    return broken
