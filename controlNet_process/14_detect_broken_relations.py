#!/usr/bin/env python3
# 14_detect_broken_relations.py
"""
14_detect_broken_relations.py (simple launcher)

Hard-coded iter: iter_000

Does NOT save anything.
- Loads iter_000/program_graph.json
- Loads sketch/target_edit/{target_label.txt, bbox_after.json}
- Patches the target node's aabb_world in-memory (edited)
- Runs:
    - same_pair detector
    - connected detector
- Prints, among ALL neighbors of the target:
    - for each neighbor: connected OK/broken (if there is a connected edge)
    - same_pair  OK/broken (if there is a same_pair edge)
"""

import os
import json
from typing import Any, Dict, List, Tuple

import numpy as np

from thresholds import (
    connect_relation_threshold_ratio,
    connect_relation_threshold_abs,
    floating_point_eps,
)

from broken_relations_type.same_pair import detect_broken_same_pairs
from broken_relations_type.connect import detect_broken_connected


# ------------------------ IO ------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def load_txt(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()


# ------------------------ Paths ------------------------

def _this_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def _iter_dir_iter000() -> str:
    return os.path.join(_this_dir(), "sketch", "dsl_optimize", "optimize_iteration", "iter_000")

def _program_graph_json(iter_dir: str) -> str:
    return os.path.join(iter_dir, "program_graph.json")

def _target_edit_dir() -> str:
    return os.path.join(_this_dir(), "sketch", "target_edit")

def _target_label_txt(target_dir: str) -> str:
    return os.path.join(target_dir, "target_label.txt")

def _edited_bbox_after_json(target_dir: str) -> str:
    return os.path.join(target_dir, "bbox_after.json")


# ------------------------ Helpers ------------------------

def _np3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)

def _compute_attach_eps_from_graph(graph: Dict[str, Any]) -> float:
    """
    Same as 12_graph_building.py:
      attach_eps = abs + ratio * median_extent
    median_extent = median over all extent entries flattened.
    """
    nodes = graph.get("nodes", [])
    if not nodes:
        return float(connect_relation_threshold_abs)

    vals: List[float] = []
    for n in nodes:
        try:
            e = _np3(n.get("extent", [0, 0, 0]))
            vals.extend([float(e[0]), float(e[1]), float(e[2])])
        except Exception:
            continue

    med = float(np.median(np.array(vals, dtype=np.float64))) if vals else 0.0
    return float(connect_relation_threshold_abs + connect_relation_threshold_ratio * max(floating_point_eps, med))

def _label_to_node(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for n in graph.get("nodes", []):
        lab = str(n.get("label", ""))
        if lab:
            out[lab] = n
    return out

def _id_to_label(graph: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for n in graph.get("nodes", []):
        try:
            out[int(n.get("id", -1))] = str(n.get("label", ""))
        except Exception:
            pass
    return out

def _set_node_aabb_world(node: Dict[str, Any], mn: np.ndarray, mx: np.ndarray) -> None:
    node["aabb_world"] = {"min": mn.tolist(), "max": mx.tolist()}

def _load_edited_aabb(bbox_after_json: str) -> Tuple[np.ndarray, np.ndarray]:
    rec = load_json(bbox_after_json)
    if not isinstance(rec, dict):
        raise RuntimeError(f"target_edit/bbox_after.json must be a dict: {bbox_after_json}")

    cand = None
    if isinstance(rec.get("edit_aabb_world", None), dict):
        cand = rec["edit_aabb_world"]
    elif isinstance(rec.get("opt_aabb_world", None), dict):
        cand = rec["opt_aabb_world"]

    if not isinstance(cand, dict) or "min" not in cand or "max" not in cand:
        raise RuntimeError("Edited bbox_after.json missing edit_aabb_world/opt_aabb_world with min/max.")

    return _np3(cand["min"]), _np3(cand["max"])

def _edges_incident_to_label(graph: Dict[str, Any], target_label: str) -> List[Dict[str, Any]]:
    out = []
    for e in graph.get("edges", []):
        if str(e.get("a_label", "")) == target_label or str(e.get("b_label", "")) == target_label:
            out.append(e)
    return out

def _neighbor_labels_from_edges(edges: List[Dict[str, Any]], target_label: str) -> List[str]:
    neigh = set()
    for e in edges:
        a = str(e.get("a_label", ""))
        b = str(e.get("b_label", ""))
        if a == target_label and b:
            neigh.add(b)
        elif b == target_label and a:
            neigh.add(a)
    return sorted(neigh)

def _key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


# ------------------------ Main ------------------------

def main() -> None:
    iter_dir = _iter_dir_iter000()
    graph_json = _program_graph_json(iter_dir)
    target_dir = _target_edit_dir()

    if not os.path.isfile(graph_json):
        raise FileNotFoundError(f"Missing program_graph.json: {graph_json} (run 12_graph_building.py first)")
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Missing target_edit dir: {target_dir} (run 13_read_changes.py first)")

    target_label = load_txt(_target_label_txt(target_dir))
    edited_mn, edited_mx = _load_edited_aabb(_edited_bbox_after_json(target_dir))

    graph = load_json(graph_json)
    if not isinstance(graph, dict) or "nodes" not in graph or "edges" not in graph:
        raise RuntimeError("program_graph.json is not in expected format.")

    attach_eps = _compute_attach_eps_from_graph(graph)
    label2node = _label_to_node(graph)
    if target_label not in label2node:
        raise RuntimeError(f"Target label '{target_label}' not in program_graph.json nodes.")

    # Apply edited AABB in-memory
    _set_node_aabb_world(label2node[target_label], edited_mn, edited_mx)

    # Run detectors (global)
    broken_same = detect_broken_same_pairs(graph, size_eps=float(floating_point_eps))
    broken_conn = detect_broken_connected(graph, attach_eps=float(attach_eps), fp_eps=float(floating_point_eps))

    # Index broken results by unordered label pair for quick lookup
    broken_same_set = set(_key(r["a_label"], r["b_label"]) for r in broken_same)
    broken_conn_set = set(_key(r["a_label"], r["b_label"]) for r in broken_conn)

    # Neighbors = all nodes that have any edge with target (same_pair, connected, prior_neighboring, etc.)
    incident_edges = _edges_incident_to_label(graph, target_label)
    neighbors = _neighbor_labels_from_edges(incident_edges, target_label)

    # Also detect what relation types exist per neighbor (so we don’t print “OK” for relations that don’t exist)
    rel_types_per_neighbor: Dict[str, set] = {nb: set() for nb in neighbors}
    for e in incident_edges:
        a = str(e.get("a_label", ""))
        b = str(e.get("b_label", ""))
        t = str(e.get("type", ""))
        other = b if a == target_label else a if b == target_label else ""
        if other in rel_types_per_neighbor:
            rel_types_per_neighbor[other].add(t)

    print()
    print("[BROKEN] target_label:", target_label)
    print("[BROKEN] attach_eps   :", float(attach_eps))
    print()

    if not neighbors:
        print("[BROKEN] No neighbors found for target in graph edges.")
        return

    for nb in neighbors:
        types = rel_types_per_neighbor.get(nb, set())
        pair = _key(target_label, nb)

        # connected status only if there is a connected edge
        if "connected" in types:
            conn_status = "BROKEN" if pair in broken_conn_set else "OK"
        else:
            conn_status = "-"

        # same_pair status only if there is a same_pair edge
        if "same_pair" in types:
            same_status = "BROKEN" if pair in broken_same_set else "OK"
        else:
            same_status = "-"

        # nice compact line
        print(f"[NEIGHBOR] {nb:30s}  connected={conn_status:6s}  same_pair={same_status:6s}  edge_types={sorted(list(types))}")

    print()
    print(f"[SUMMARY] broken_same_pair={len(broken_same)}  broken_connected={len(broken_conn)}")


if __name__ == "__main__":
    main()
