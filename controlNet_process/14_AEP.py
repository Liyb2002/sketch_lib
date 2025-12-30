#!/usr/bin/env python3
# 14_AEP.py
"""
14_AEP.py

AEP (part 1): decide which neighbors need edits.

Hard-coded iter: iter_000

Steps:
1) Read the edit (target_label + before/after AABB) from sketch/target_edit/
2) Load program_graph.json and find all neighbors (incident edges).
3) If neighbor relation is "symmetry" (we treat edge type == "same_pair" as symmetry):
     print "to change!"
4) If neighbor relation is "connect" (edge type == "connected"):
     call AEP/connect_change.py to recompute anchor points (pin midpoint between closest points).
     If anchor changes -> neighbor needs edit.
     Print old/new anchors if changed; else print "no change".
"""

import os
import json
from typing import Any, Dict, List, Tuple

import numpy as np

from thresholds import floating_point_eps
from AEP.connect_change import compute_anchor_change


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

def _bbox_before_json(target_dir: str) -> str:
    return os.path.join(target_dir, "bbox_before.json")

def _bbox_after_json(target_dir: str) -> str:
    return os.path.join(target_dir, "bbox_after.json")


# ------------------------ Helpers ------------------------

def _np3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)

def _node_by_label(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for n in graph.get("nodes", []):
        lab = str(n.get("label", ""))
        if lab:
            out[lab] = n
    return out

def _node_aabb(node: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    aabb = node.get("aabb_world", {}) if isinstance(node.get("aabb_world", {}), dict) else {}
    mn = _np3(aabb.get("min", [0, 0, 0]))
    mx = _np3(aabb.get("max", [0, 0, 0]))
    return mn, mx

def _edited_aabb_from_rec(rec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accept either:
      rec["edit_aabb_world"] or rec["opt_aabb_world"]
    """
    cand = None
    if isinstance(rec.get("edit_aabb_world", None), dict):
        cand = rec["edit_aabb_world"]
    elif isinstance(rec.get("opt_aabb_world", None), dict):
        cand = rec["opt_aabb_world"]

    if not isinstance(cand, dict) or "min" not in cand or "max" not in cand:
        raise RuntimeError("bbox_after.json missing edit_aabb_world/opt_aabb_world with min/max")

    return _np3(cand["min"]), _np3(cand["max"])

def _before_aabb_from_rec_or_graph(before_rec: Dict[str, Any], graph_node: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prefer bbox_before.json's aabb if present; else fallback to graph node aabb_world.
    """
    cand = None
    if isinstance(before_rec.get("edit_aabb_world", None), dict):
        cand = before_rec["edit_aabb_world"]
    elif isinstance(before_rec.get("opt_aabb_world", None), dict):
        cand = before_rec["opt_aabb_world"]

    if isinstance(cand, dict) and "min" in cand and "max" in cand:
        return _np3(cand["min"]), _np3(cand["max"])

    return _node_aabb(graph_node)

def _incident_edges(graph: Dict[str, Any], target_label: str) -> List[Dict[str, Any]]:
    out = []
    for e in graph.get("edges", []):
        if str(e.get("a_label", "")) == target_label or str(e.get("b_label", "")) == target_label:
            out.append(e)
    return out

def _other_label(edge: Dict[str, Any], target_label: str) -> str:
    a = str(edge.get("a_label", ""))
    b = str(edge.get("b_label", ""))
    if a == target_label:
        return b
    if b == target_label:
        return a
    return ""


# ------------------------ Main ------------------------

def main() -> None:
    iter_dir = _iter_dir_iter000()
    graph_path = _program_graph_json(iter_dir)
    target_dir = _target_edit_dir()

    if not os.path.isfile(graph_path):
        raise FileNotFoundError(f"Missing program_graph.json: {graph_path} (run 12_graph_building.py first)")
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Missing target_edit folder: {target_dir} (run 13_read_changes.py first)")

    target_label = load_txt(_target_label_txt(target_dir))

    before_path = _bbox_before_json(target_dir)
    after_path = _bbox_after_json(target_dir)
    if not os.path.isfile(after_path):
        raise FileNotFoundError(f"Missing: {after_path}")

    before_rec = load_json(before_path) if os.path.isfile(before_path) else {}
    after_rec = load_json(after_path)
    mnA_after, mxA_after = _edited_aabb_from_rec(after_rec)

    graph = load_json(graph_path)
    label2node = _node_by_label(graph)
    if target_label not in label2node:
        raise RuntimeError(f"Target label '{target_label}' not found in graph nodes.")

    # target BEFORE AABB (needed to compare anchors)
    mnA_before, mxA_before = _before_aabb_from_rec_or_graph(before_rec, label2node[target_label])

    edges = _incident_edges(graph, target_label)
    if not edges:
        print("[AEP] target has no incident edges:", target_label)
        return

    print("[AEP] iter_dir     :", os.path.abspath(iter_dir))
    print("[AEP] graph        :", os.path.abspath(graph_path))
    print("[AEP] target_label :", target_label)
    print()

    # Iterate neighbors by edges
    for e in edges:
        et = str(e.get("type", ""))
        nb = _other_label(e, target_label)
        if not nb:
            continue

        # Treat "same_pair" as "symmetry" per your instruction
        if et == "same_pair":
            print(f"[SYM] {target_label}  <->  {nb} : to change!")
            continue

        if et == "connected":
            if nb not in label2node:
                print(f"[CONN] {target_label} <-> {nb} : neighbor missing in nodes (skip)")
                continue

            mnB, mxB = _node_aabb(label2node[nb])

            changed, old_pin, new_pin = compute_anchor_change(
                mnA_before=mnA_before,
                mxA_before=mxA_before,
                mnA_after=mnA_after,
                mxA_after=mxA_after,
                mnB=mnB,
                mxB=mxB,
                eps=float(floating_point_eps),
            )

            if changed:
                print(f"[CONN] {target_label}  <->  {nb} : to change!")
                print("       old_anchor:", old_pin.tolist())
                print("       new_anchor:", new_pin.tolist())
            else:
                print(f"[CONN] {target_label}  <->  {nb} : no change")

            continue

        # ignore other edge types
        # (prior_neighboring etc.)
        # print(f"[SKIP] edge type={et} {target_label}<->{nb}")

    print()


if __name__ == "__main__":
    main()
