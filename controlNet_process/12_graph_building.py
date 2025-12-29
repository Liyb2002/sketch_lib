# 12_graph_building.py  (updated: loads thresholds from graph_building/thresholds.py)
#!/usr/bin/env python3
"""
12_graph_building.py

Build a simple component graph from optimized bounding boxes.

Hard-coded iter:
- sketch/dsl_optimize/optimize_iteration/iter_000/

Outputs:
- sketch/dsl_optimize/optimize_iteration/iter_000/program_graph.json

Also calls visualization:
- per-label screenshots to:
    sketch/dsl_optimize/optimize_iteration/iter_000/graph_vis/<label>.png
(overlay bboxes + attachment pins on sketch/3d_reconstruction/fused_model.ply)
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np

# ------------------------ Thresholds ------------------------
# Only change: load relation thresholds from thresholds.py
try:
    from graph_building.thresholds import (
        same_pair_relation_threshold_confidence,
        connect_relation_threshold_ratio,
        connect_relation_threshold_abs,
        floating_point_eps,
    )
except Exception:
    # Fallback to preserve old behavior if thresholds.py is missing
    same_pair_relation_threshold_confidence = 0.0
    connect_relation_threshold_ratio = 0.02
    connect_relation_threshold_abs = 0.0
    floating_point_eps = 1e-12


# ------------------------ IO ------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def infer_paths_from_iter_dir(iter_dir: str) -> Tuple[str, str]:
    """
    iter_dir:
      .../sketch/dsl_optimize/optimize_iteration/iter_XXX
    returns:
      optimize_results_dir, relations_json
    """
    iter_dir = os.path.abspath(iter_dir)
    optimize_results_dir = os.path.join(iter_dir, "optimize_results")

    marker = os.sep + "sketch" + os.sep
    idx = iter_dir.rfind(marker)
    if idx < 0:
        relations_json = os.path.join(os.getcwd(), "sketch", "dsl_optimize", "relations.json")
    else:
        sketch_root = iter_dir[: idx + len(marker)]  # ends with ".../sketch/"
        relations_json = os.path.join(sketch_root, "dsl_optimize", "relations.json")

    return optimize_results_dir, relations_json


# ------------------------ Geometry ------------------------

@dataclass
class Node:
    label: str
    center: np.ndarray  # (3,)
    extent: np.ndarray  # (3,)
    R: np.ndarray       # (3,3)
    aabb_min: np.ndarray  # (3,)
    aabb_max: np.ndarray  # (3,)

def _obb_to_world_aabb(center: np.ndarray, R: np.ndarray, extent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust AABB for an OBB: transform 8 corners.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    extent = np.asarray(extent, dtype=np.float64).reshape(3)

    half = 0.5 * extent
    xs = [-half[0], half[0]]
    ys = [-half[1], half[1]]
    zs = [-half[2], half[2]]
    corners_local = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    corners_world = (R @ corners_local.T).T + center[None, :]
    return corners_world.min(axis=0), corners_world.max(axis=0)

def _aabb_overlap_1d(mn1: float, mx1: float, mn2: float, mx2: float) -> float:
    return float(max(0.0, min(mx1, mx2) - max(mn1, mn2)))

def _aabb_overlap_volume(mn1: np.ndarray, mx1: np.ndarray, mn2: np.ndarray, mx2: np.ndarray) -> float:
    ox = _aabb_overlap_1d(mn1[0], mx1[0], mn2[0], mx2[0])
    oy = _aabb_overlap_1d(mn1[1], mx1[1], mn2[1], mx2[1])
    oz = _aabb_overlap_1d(mn1[2], mx1[2], mn2[2], mx2[2])
    return float(ox * oy * oz)

def _aabb_gaps(mn1: np.ndarray, mx1: np.ndarray, mn2: np.ndarray, mx2: np.ndarray) -> np.ndarray:
    """
    Per-axis separation distance between boxes (0 means overlapping/touching along that axis).
    """
    g = np.zeros((3,), dtype=np.float64)
    for k in range(3):
        g[k] = max(0.0, float(max(mn2[k] - mx1[k], mn1[k] - mx2[k])))
    return g

def _contains_aabb(mn_outer: np.ndarray, mx_outer: np.ndarray, mn_inner: np.ndarray, mx_inner: np.ndarray, eps: float) -> bool:
    return bool(np.all(mn_inner >= mn_outer - eps) and np.all(mx_inner <= mx_outer + eps))

def _contact_axis_and_type(
    mn1: np.ndarray,
    mx1: np.ndarray,
    mn2: np.ndarray,
    mx2: np.ndarray,
    attach_eps: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Decide if connected + classify attachment.
    """
    gaps = _aabb_gaps(mn1, mx1, mn2, mx2)
    max_gap = float(np.max(gaps))

    overlap_vol = _aabb_overlap_volume(mn1, mx1, mn2, mx2)
    overlaps = np.array([
        _aabb_overlap_1d(mn1[0], mx1[0], mn2[0], mx2[0]),
        _aabb_overlap_1d(mn1[1], mx1[1], mn2[1], mx2[1]),
        _aabb_overlap_1d(mn1[2], mx1[2], mn2[2], mx2[2]),
    ], dtype=np.float64)

    info: Dict[str, Any] = {
        "gaps": gaps.tolist(),
        "max_gap": max_gap,
        "overlap_volume": float(overlap_vol),
        "overlaps": overlaps.tolist(),
    }

    connected = (overlap_vol > 0.0) or (max_gap <= float(attach_eps))
    if not connected:
        return False, "none", info

    if _contains_aabb(mn1, mx1, mn2, mx2, eps=attach_eps):
        return True, "contains(A_contains_B)", info
    if _contains_aabb(mn2, mx2, mn1, mx1, eps=attach_eps):
        return True, "contains(B_contains_A)", info

    if overlap_vol > 0.0:
        return True, "overlap_penetration", info

    near_axes = (gaps <= float(attach_eps) + float(floating_point_eps)).astype(np.int32)
    near_count = int(np.sum(near_axes))

    if near_count == 1:
        axis = int(np.argmax(near_axes))
        return True, f"face_touch(axis={axis})", info
    if near_count == 2:
        axes = np.where(near_axes > 0)[0].tolist()
        return True, f"edge_touch(axes={axes})", info
    if near_count >= 3:
        return True, "point_touch", info

    return True, "near", info


# ------------------------ Load nodes ------------------------

def load_nodes_from_optimize_results(optimize_results_dir: str) -> List[Node]:
    if not os.path.isdir(optimize_results_dir):
        raise FileNotFoundError(f"Missing optimize_results_dir: {optimize_results_dir}")

    nodes: List[Node] = []
    for name in sorted(os.listdir(optimize_results_dir)):
        label_dir = os.path.join(optimize_results_dir, name)
        if not os.path.isdir(label_dir):
            continue
        after_json = os.path.join(label_dir, "bbox_after.json")
        if not os.path.isfile(after_json):
            continue

        rec = load_json(after_json)
        label = str(rec.get("label", name))
        obb = rec.get("obb", {})

        center = np.asarray(obb.get("center", [0, 0, 0]), dtype=np.float64).reshape(3)
        extent = np.asarray(obb.get("extent", [0, 0, 0]), dtype=np.float64).reshape(3)
        R = np.asarray(obb.get("R", np.eye(3).tolist()), dtype=np.float64).reshape(3, 3)

        opt_aabb = rec.get("opt_aabb_world", None)
        if isinstance(opt_aabb, dict) and "min" in opt_aabb and "max" in opt_aabb:
            aabb_min = np.asarray(opt_aabb["min"], dtype=np.float64).reshape(3)
            aabb_max = np.asarray(opt_aabb["max"], dtype=np.float64).reshape(3)
        else:
            aabb_min, aabb_max = _obb_to_world_aabb(center, R, extent)

        nodes.append(Node(
            label=label,
            center=center,
            extent=extent,
            R=R,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        ))

    if not nodes:
        raise RuntimeError(f"No nodes found under: {optimize_results_dir}")

    return nodes


# ------------------------ Graph build ------------------------

def build_graph(
    nodes: List[Node],
    relations_json: str,
    # defaults now come from thresholds.py (but caller can still override)
    attach_eps_ratio: float = None,
    attach_eps_abs: float = None,
) -> Dict[str, Any]:
    # Only change: resolve defaults from thresholds.py
    if attach_eps_ratio is None:
        attach_eps_ratio = float(connect_relation_threshold_ratio)
    if attach_eps_abs is None:
        attach_eps_abs = float(connect_relation_threshold_abs)

    extents = np.array([n.extent for n in nodes], dtype=np.float64)
    med = float(np.median(extents))
    attach_eps = float(attach_eps_abs + attach_eps_ratio * max(float(floating_point_eps), med))

    rel = load_json(relations_json) if os.path.isfile(relations_json) else {}
    same_pairs = rel.get("same_pairs", []) if isinstance(rel, dict) else []
    neighboring_pairs = rel.get("neighboring_pairs", []) if isinstance(rel, dict) else []

    node_table = []
    label_to_idx = {}
    for i, n in enumerate(nodes):
        label_to_idx[n.label] = i
        node_table.append({
            "id": int(i),
            "label": n.label,
            "center": n.center.tolist(),
            "extent": n.extent.tolist(),
            "R": n.R.tolist(),
            "aabb_world": {"min": n.aabb_min.tolist(), "max": n.aabb_max.tolist()},
        })

    edges = []
    N = len(nodes)
    for i in range(N):
        for j in range(i + 1, N):
            A = nodes[i]
            B = nodes[j]
            connected, attach_type, info = _contact_axis_and_type(
                A.aabb_min, A.aabb_max, B.aabb_min, B.aabb_max, attach_eps=attach_eps
            )
            if not connected:
                continue
            edges.append({
                "type": "connected",
                "a": int(i),
                "b": int(j),
                "a_label": A.label,
                "b_label": B.label,
                "attachment": attach_type,
                "metrics": info,
            })

    same_edges = []
    for sp in same_pairs:
        # Only change: threshold read from thresholds.py (default 0.0 preserves old behavior)
        if float(sp.get("confidence", 1.0)) < float(same_pair_relation_threshold_confidence):
            continue

        a = str(sp.get("a", ""))
        b = str(sp.get("b", ""))
        if a in label_to_idx and b in label_to_idx:
            same_edges.append({
                "type": "same_pair",
                "a": int(label_to_idx[a]),
                "b": int(label_to_idx[b]),
                "a_label": a,
                "b_label": b,
                "confidence": float(sp.get("confidence", 1.0)),
                "evidence": sp.get("evidence", ""),
            })

    neighbor_edges = []
    for nb in neighboring_pairs:
        a = str(nb.get("a", ""))
        b = str(nb.get("b", ""))
        if a in label_to_idx and b in label_to_idx:
            neighbor_edges.append({
                "type": "prior_neighboring",
                "a": int(label_to_idx[a]),
                "b": int(label_to_idx[b]),
                "a_label": a,
                "b_label": b,
                "confidence": float(nb.get("confidence", 0.0)),
                "evidence": nb.get("evidence", ""),
            })

    graph = {
        "meta": {
            "attach_eps": float(attach_eps),
            "attach_eps_ratio": float(attach_eps_ratio),
            "attach_eps_abs": float(attach_eps_abs),
            "relations_json": os.path.abspath(relations_json) if relations_json else "",
            "thresholds": {
                "same_pair_relation_threshold_confidence": float(same_pair_relation_threshold_confidence),
                "connect_relation_threshold_ratio": float(connect_relation_threshold_ratio),
                "connect_relation_threshold_abs": float(connect_relation_threshold_abs),
                "floating_point_eps": float(floating_point_eps),
            },
        },
        "nodes": node_table,
        "edges": edges + same_edges + neighbor_edges,
        "same_pairs": same_pairs,
        "neighboring_pairs": neighboring_pairs,
    }
    return graph


# ------------------------ CLI ------------------------

def main():
    # Hard-code iter_000 relative to this script's directory.
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    iter_dir = os.path.join(
        THIS_DIR,
        "sketch",
        "dsl_optimize",
        "optimize_iteration",
        "iter_000",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", type=str, default="", help="Optional output json path.")

    # Only change: defaults now come from thresholds.py; keep flags for override
    ap.add_argument("--attach_eps_ratio", type=float, default=None)
    ap.add_argument("--attach_eps_abs", type=float, default=None)

    # visualization options (called by default)
    ap.add_argument("--no_vis", action="store_true", help="Skip visualization call.")
    ap.add_argument("--vis_only_label", type=str, default="", help="Only visualize this label.")
    ap.add_argument("--vis_max_labels", type=int, default=0, help="Limit number of labels rendered (0 = all).")
    ap.add_argument("--vis_pin_radius_ratio", type=float, default=0.01, help="Pin radius ratio w.r.t. median extent.")
    args = ap.parse_args()

    optimize_results_dir, relations_json = infer_paths_from_iter_dir(iter_dir)

    # Only change: resolve and print the thresholds we will use
    attach_eps_ratio = float(connect_relation_threshold_ratio) if args.attach_eps_ratio is None else float(args.attach_eps_ratio)
    attach_eps_abs = float(connect_relation_threshold_abs) if args.attach_eps_abs is None else float(args.attach_eps_abs)

    print("[GRAPH] iter_dir            :", os.path.abspath(iter_dir))
    print("[GRAPH] optimize_results_dir:", os.path.abspath(optimize_results_dir))
    print("[GRAPH] relations_json      :", os.path.abspath(relations_json))
    print("[GRAPH] connect_relation_threshold_ratio:", attach_eps_ratio)
    print("[GRAPH] connect_relation_threshold_abs  :", attach_eps_abs)
    print("[GRAPH] same_pair_relation_threshold_confidence:", float(same_pair_relation_threshold_confidence))

    nodes = load_nodes_from_optimize_results(optimize_results_dir)
    graph = build_graph(
        nodes=nodes,
        relations_json=relations_json,
        attach_eps_ratio=attach_eps_ratio,
        attach_eps_abs=attach_eps_abs,
    )

    out_json = args.out_json.strip() or os.path.join(os.path.abspath(iter_dir), "program_graph.json")
    save_json(out_json, graph)

    conn_edges = [e for e in graph["edges"] if e.get("type") == "connected"]
    same_edges = [e for e in graph["edges"] if e.get("type") == "same_pair"]
    neigh_edges = [e for e in graph["edges"] if e.get("type") == "prior_neighboring"]

    print(f"[GRAPH] nodes              : {len(graph['nodes'])}")
    print(f"[GRAPH] connected edges    : {len(conn_edges)}")
    print(f"[GRAPH] same_pair edges    : {len(same_edges)}")
    print(f"[GRAPH] prior_neighbor edges: {len(neigh_edges)}")
    print(f"[GRAPH] wrote: {out_json}")

    if not args.no_vis:
        try:
            from graph_building.vis import run_graph_vis
            # run_graph_vis(
            #     iter_dir=os.path.abspath(iter_dir),
            #     graph_json=os.path.abspath(out_json),
            #     only_label=args.vis_only_label.strip(),
            #     max_labels=int(args.vis_max_labels),
            #     pin_radius_ratio=float(args.vis_pin_radius_ratio),
            #     caller_file=__file__,  # <-- IMPORTANT: makes fused_model.ply relative to 12_graph_building.py
            # )

        except Exception as ex:
            print("[GRAPH] visualization failed:", str(ex))


if __name__ == "__main__":
    main()
