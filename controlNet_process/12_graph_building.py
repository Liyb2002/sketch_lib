#!/usr/bin/env python3
"""
12_graph_building.py

Build a simple component graph from optimized bounding boxes.

Inputs (hard-coded iter_000):
- Iter folder:
    sketch/dsl_optimize/optimize_iteration/iter_000/
      optimize_results/<label>/bbox_after.json
      optimize_results/<label>/heat_map_<label>.ply   (present, not used here)
- relations.json:
    sketch/dsl_optimize/relations.json

Outputs:
- Writes a graph json to:
    sketch/dsl_optimize/optimize_iteration/iter_000/program_graph.json

Graph nodes:
- one node per component label
- stores bbox center/extent/R and world AABB

Edges:
1) connectivity: inferred from AABB proximity / contact (not program tokens yet)
2) attachment: if connected, describe how (face-touch axis, overlap/penetration, containment)
3) same_pairs copied from relations.json, also mirrored as edges

How connectivity is inferred (robust + easy to tune):
- Compute WORLD AABB for each node.
- Define per-axis gap:
    gap = max(0, max(mnB - mxA, mnA - mxB))
- If gaps are all ~0 => overlap/penetration
- Else if max gap <= attach_eps => "touch/near"
- Else not connected.

Attachment classification:
- containment: one AABB fully inside another (within eps)
- overlap: overlap volume > 0 (penetration)
- face_touch: near along one axis, overlap on the other two axes
- edge_touch: near along two axes, overlap on one axis
- point_touch: near along three axes (rare)
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np


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

    # relations.json sits at: .../sketch/dsl_optimize/relations.json
    # Find ".../sketch/" marker then append dsl_optimize/relations.json
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

    # Connected if overlap (penetration) OR near-touch within attach_eps
    connected = (overlap_vol > 0.0) or (max_gap <= float(attach_eps))
    if not connected:
        return False, "none", info

    # Containment (dominant relation)
    if _contains_aabb(mn1, mx1, mn2, mx2, eps=attach_eps):
        return True, "contains(A_contains_B)", info
    if _contains_aabb(mn2, mx2, mn1, mx1, eps=attach_eps):
        return True, "contains(B_contains_A)", info

    # Penetration / overlap
    if overlap_vol > 0.0:
        return True, "overlap_penetration", info

    # Touch / near: determine touch dimensionality by counting small gaps
    near_axes = (gaps <= float(attach_eps) + 1e-12).astype(np.int32)
    near_count = int(np.sum(near_axes))

    # face touch: near along 1 axis
    # edge touch: near along 2 axes
    # point touch: near along 3 axes
    if near_count == 1:
        axis = int(np.argmax(near_axes))
        return True, f"face_touch(axis={axis})", info
    if near_count == 2:
        axes = np.where(near_axes > 0)[0].tolist()
        return True, f"edge_touch(axes={axes})", info
    if near_count >= 3:
        return True, "point_touch", info

    # fallback
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

        # Prefer opt_aabb_world if present (optimizer writes it); else compute from OBB
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
    attach_eps_ratio: float = 0.02,
    attach_eps_abs: float = 0.0,
) -> Dict[str, Any]:
    """
    attach_eps:
      default derived from scene scale: attach_eps_ratio * median_extent
      plus optional absolute epsilon (attach_eps_abs)
    """
    # scene scale (median extent length)
    extents = np.array([n.extent for n in nodes], dtype=np.float64)
    med = float(np.median(extents))
    attach_eps = float(attach_eps_abs + attach_eps_ratio * max(1e-12, med))

    # copy same_pairs from relations.json
    rel = load_json(relations_json) if os.path.isfile(relations_json) else {}
    same_pairs = rel.get("same_pairs", []) if isinstance(rel, dict) else []
    neighboring_pairs = rel.get("neighboring_pairs", []) if isinstance(rel, dict) else []

    # nodes table
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

    # inferred connectivity edges
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

    # same_pairs edges (copied)
    same_edges = []
    for sp in same_pairs:
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

    # neighboring_pairs edges (copied, if you want them in the graph too)
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
        },
        "nodes": node_table,
        "edges": edges + same_edges + neighbor_edges,
        "same_pairs": same_pairs,                 # copied verbatim
        "neighboring_pairs": neighboring_pairs,   # copied verbatim
    }
    return graph


# ------------------------ CLI ------------------------

def main():
    # Hard-code iter_000 relative to this script's directory.
    # Assumes this script is run from repo root OR that this file lives where sketch/ is reachable.
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    iter_dir = os.path.join(
        THIS_DIR,
        "sketch",
        "dsl_optimize",
        "optimize_iteration",
        "iter_000",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Optional output json path. Default: <iter_dir>/program_graph.json",
    )
    ap.add_argument(
        "--attach_eps_ratio",
        type=float,
        default=0.02,
        help="Connectivity epsilon as ratio of median bbox extent (default 0.02)",
    )
    ap.add_argument(
        "--attach_eps_abs",
        type=float,
        default=0.0,
        help="Additional absolute epsilon in world units (default 0.0)",
    )
    args = ap.parse_args()

    optimize_results_dir, relations_json = infer_paths_from_iter_dir(iter_dir)
    print("[GRAPH] iter_dir            :", os.path.abspath(iter_dir))
    print("[GRAPH] optimize_results_dir:", os.path.abspath(optimize_results_dir))
    print("[GRAPH] relations_json      :", os.path.abspath(relations_json))
    print("[GRAPH] attach_eps_ratio    :", args.attach_eps_ratio)
    print("[GRAPH] attach_eps_abs      :", args.attach_eps_abs)

    nodes = load_nodes_from_optimize_results(optimize_results_dir)
    graph = build_graph(
        nodes=nodes,
        relations_json=relations_json,
        attach_eps_ratio=float(args.attach_eps_ratio),
        attach_eps_abs=float(args.attach_eps_abs),
    )

    out_json = args.out_json.strip() or os.path.join(os.path.abspath(iter_dir), "program_graph.json")
    save_json(out_json, graph)

    # quick summary
    conn_edges = [e for e in graph["edges"] if e.get("type") == "connected"]
    same_edges = [e for e in graph["edges"] if e.get("type") == "same_pair"]
    neigh_edges = [e for e in graph["edges"] if e.get("type") == "prior_neighboring"]

    print(f"[GRAPH] nodes            : {len(graph['nodes'])}")
    print(f"[GRAPH] connected edges  : {len(conn_edges)}")
    print(f"[GRAPH] same_pair edges  : {len(same_edges)}")
    print(f"[GRAPH] prior_neighbor edges: {len(neigh_edges)}")
    print(f"[GRAPH] wrote: {out_json}")


if __name__ == "__main__":
    main()
