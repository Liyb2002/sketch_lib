#!/usr/bin/env python3
# 14_AEP.py
"""
14_AEP.py

AEP (simple propagation, translation-only) + visualization.

Hard-coded iter: iter_000

Behavior:
- Read target edit (target_label + AABB before/after) from sketch/target_edit/
- Load iter_000/program_graph.json
- For each neighbor edge incident to target:
  - symmetry (type == "same_pair"): print "to change!"
  - connect (type == "connected"):
      - compute anchor change using (A_before,B_before) vs (A_after,B_before)
      - if anchor changed:
          - neighbor is marked "changed"
          - apply translation-only edit to neighbor B:
              delta = center(A_after) - center(A_before)
              B_after = B_before + delta
      - else: neighbor unchanged

Finally:
- Open 2 visualization windows (before / after) showing:
  - target in red
  - neighbors: blue
  - after: unchanged neighbors faint blue, changed neighbors dark blue, target red
  - overlays fused_model.ply as faint gray
"""

import os
import json
from typing import Any, Dict, List, Tuple, Set

import numpy as np

from thresholds import floating_point_eps
from AEP.connect_detect import compute_anchor_change
from AEP.connect_neighbor_edits import translate_neighbor_by_target_delta
from AEP.vis import show_aep_before_after


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

def _fused_model_ply() -> str:
    # same level as 14_AEP.py
    return os.path.join(_this_dir(), "sketch", "3d_reconstruction", "fused_model.ply")


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

def _set_node_aabb(node: Dict[str, Any], mn: np.ndarray, mx: np.ndarray) -> None:
    node["aabb_world"] = {"min": _np3(mn).tolist(), "max": _np3(mx).tolist()}

def _edited_aabb_from_rec(rec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    cand = None
    if isinstance(rec.get("edit_aabb_world", None), dict):
        cand = rec["edit_aabb_world"]
    elif isinstance(rec.get("opt_aabb_world", None), dict):
        cand = rec["opt_aabb_world"]
    if not isinstance(cand, dict) or "min" not in cand or "max" not in cand:
        raise RuntimeError("bbox_after.json missing edit_aabb_world/opt_aabb_world with min/max")
    return _np3(cand["min"]), _np3(cand["max"])

def _before_aabb_from_rec_or_graph(before_rec: Dict[str, Any], graph_node: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
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

    graph = load_json(graph_path)
    label2node = _node_by_label(graph)
    if target_label not in label2node:
        raise RuntimeError(f"Target label '{target_label}' not found in graph nodes.")

    # AABB for A before/after
    mnA_before, mxA_before = _before_aabb_from_rec_or_graph(before_rec, label2node[target_label])
    mnA_after,  mxA_after  = _edited_aabb_from_rec(after_rec)

    edges = _incident_edges(graph, target_label)
    if not edges:
        print("[AEP] target has no incident edges:", target_label)
        return

    # Snapshot BEFORE AABBs for target + its neighbors
    before_aabbs: Dict[str, Dict[str, List[float]]] = {}
    after_aabbs: Dict[str, Dict[str, List[float]]] = {}
    changed_neighbors: Set[str] = set()

    # include target
    before_aabbs[target_label] = {"min": mnA_before.tolist(), "max": mxA_before.tolist()}
    after_aabbs[target_label]  = {"min": mnA_after.tolist(),  "max": mxA_after.tolist()}

    # Gather neighbor labels first (so vis includes all even if we skip some edges)
    neighbor_labels = []
    for e in edges:
        nb = _other_label(e, target_label)
        if nb and nb in label2node and nb not in neighbor_labels:
            neighbor_labels.append(nb)

    for nb in neighbor_labels:
        mnB, mxB = _node_aabb(label2node[nb])
        before_aabbs[nb] = {"min": mnB.tolist(), "max": mxB.tolist()}
        after_aabbs[nb]  = {"min": mnB.tolist(), "max": mxB.tolist()}  # default unchanged

    print("[AEP] iter_dir     :", os.path.abspath(iter_dir))
    print("[AEP] graph        :", os.path.abspath(graph_path))
    print("[AEP] target_label :", target_label)
    print()

    for e in edges:
        
        et = str(e.get("type", ""))
        nb = _other_label(e, target_label)
        if not nb or nb not in label2node:
            continue
    
        # symmetry
        if et == "same_pair":
            print(f"[SYM]  {target_label}  <->  {nb} : to change!")

            # Apply RESIZE-only same_pair edit (orientation-aware axis mapping)
            from AEP.same_pair_neighbor_edits import resize_same_pair_neighbor

            if nb not in label2node:
                print(f"       [SYM] neighbor missing in nodes (skip)")
                continue

            # Use current stored neighbor aabb as "before"
            mnB_before, mxB_before = _node_aabb(label2node[nb])

            mnB_after, mxB_after, size_deltaB, axis_map = resize_same_pair_neighbor(
                mnA_before=mnA_before,
                mxA_before=mxA_before,
                mnA_after=mnA_after,
                mxA_after=mxA_after,
                mnB=mnB_before,
                mxB=mxB_before,
                eps=float(floating_point_eps),
            )

            changed_neighbors.add(nb)
            after_aabbs[nb] = {"min": mnB_after.tolist(), "max": mxB_after.tolist()}

            print("       same_pair_axis_map (B_axis -> A_axis):", axis_map)
            print("       same_pair_size_delta:", size_deltaB.tolist())
            print("       B_before_aabb  :", {"min": mnB_before.tolist(), "max": mxB_before.tolist()})
            print("       B_after_aabb   :", {"min": mnB_after.tolist(),  "max": mxB_after.tolist()})
            continue

        # connected
        if et == "connected":
            mnB_before, mxB_before = _node_aabb(label2node[nb])

            changed, old_pin, new_pin_tmp = compute_anchor_change(
                mnA_before=mnA_before,
                mxA_before=mxA_before,
                mnA_after=mnA_after,
                mxA_after=mxA_after,
                mnB=mnB_before,
                mxB=mxB_before,
                eps=float(floating_point_eps),
            )

            if not changed:
                print(f"[CONN] {target_label}  <->  {nb} : no change")
                continue

            print(f"[CONN] {target_label}  <->  {nb} : to change!")
            print("       old_anchor(A_before,B_before):", old_pin.tolist())
            print("       new_anchor(A_after,B_before) :", new_pin_tmp.tolist())

            mnB_after, mxB_after, delta = translate_neighbor_by_target_delta(
                mnA_before=mnA_before,
                mxA_before=mxA_before,
                mnA_after=mnA_after,
                mxA_after=mxA_after,
                mnB=mnB_before,
                mxB=mxB_before,
            )

            changed_neighbors.add(nb)
            after_aabbs[nb] = {"min": mnB_after.tolist(), "max": mxB_after.tolist()}

            print("       neighbor_delta:", delta.tolist())
            print("       B_before_aabb :", {"min": mnB_before.tolist(), "max": mxB_before.tolist()})
            print("       B_after_aabb  :", {"min": mnB_after.tolist(),  "max": mxB_after.tolist()})
            continue

    print()
    # Visualization
    fused_ply = _fused_model_ply()
    print("[AEP][VIS] fused_model.ply:", os.path.abspath(fused_ply))
    show_aep_before_after(
        fused_ply_path=fused_ply,
        target_label=target_label,
        before_aabbs=before_aabbs,
        after_aabbs=after_aabbs,
        changed_neighbors=changed_neighbors,
    )


    # ------------------------ Save per-label AABBs (all labels) ------------------------

    out_path = os.path.join(target_dir, "all_labels_aabbs.json")

    # labels whose bbox changed in this run (target always counts as changed)
    changed_labels = set(changed_neighbors)
    changed_labels.add(target_label)

    out = {
        "target_label": target_label,
        "changed_labels": sorted(list(changed_labels)),
        "labels": {}
    }

    # Save for every label we have in before_aabbs (target + all incident neighbors)
    for lab in sorted(before_aabbs.keys()):
        b = before_aabbs[lab]  # {"min": [...], "max": [...]}
        a = after_aabbs.get(lab, b)

        if lab in changed_labels:
            out["labels"][lab] = {
                "before": b,
                "after": a,
            }
        else:
            out["labels"][lab] = {
                "aabb": b
            }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("[AEP][SAVE] per-label AABBs:", os.path.abspath(out_path))


if __name__ == "__main__":
    main()
