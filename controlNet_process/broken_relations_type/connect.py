# broken_relations_type/connect.py
"""
Broken connected detection.

User rule:
- connected fails if we can no longer detect connect.

We use the SAME connectivity criterion as graph_building:
- Compute WORLD AABB gaps (per-axis separation).
- Connected if (overlap_volume > 0) OR (max_gap <= attach_eps)

We only check edges that are already in the graph with type == "connected".
If an edge existed before but now fails this test => broken.
"""

from typing import Any, Dict, List, Tuple
import numpy as np


def _np3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)

def _aabb_overlap_1d(mn1: float, mx1: float, mn2: float, mx2: float) -> float:
    return float(max(0.0, min(mx1, mx2) - max(mn1, mn2)))

def _aabb_overlap_volume(mn1: np.ndarray, mx1: np.ndarray, mn2: np.ndarray, mx2: np.ndarray) -> float:
    ox = _aabb_overlap_1d(mn1[0], mx1[0], mn2[0], mx2[0])
    oy = _aabb_overlap_1d(mn1[1], mx1[1], mn2[1], mx2[1])
    oz = _aabb_overlap_1d(mn1[2], mx1[2], mn2[2], mx2[2])
    return float(ox * oy * oz)

def _aabb_gaps(mn1: np.ndarray, mx1: np.ndarray, mn2: np.ndarray, mx2: np.ndarray) -> np.ndarray:
    g = np.zeros((3,), dtype=np.float64)
    for k in range(3):
        g[k] = max(0.0, float(max(mn2[k] - mx1[k], mn1[k] - mx2[k])))
    return g

def _is_connected(mn1: np.ndarray, mx1: np.ndarray, mn2: np.ndarray, mx2: np.ndarray, attach_eps: float) -> Tuple[bool, Dict[str, Any]]:
    gaps = _aabb_gaps(mn1, mx1, mn2, mx2)
    max_gap = float(np.max(gaps))

    overlap_vol = _aabb_overlap_volume(mn1, mx1, mn2, mx2)
    connected = (overlap_vol > 0.0) or (max_gap <= float(attach_eps))

    info = {
        "gaps": gaps.tolist(),
        "max_gap": float(max_gap),
        "overlap_volume": float(overlap_vol),
        "attach_eps": float(attach_eps),
    }
    return bool(connected), info

def _id_to_node(graph: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for n in graph.get("nodes", []):
        try:
            out[int(n.get("id", -1))] = n
        except Exception:
            pass
    return out

def _node_aabb(node: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    aabb = node.get("aabb_world", {}) if isinstance(node.get("aabb_world", {}), dict) else {}
    mn = _np3(aabb.get("min", [0, 0, 0]))
    mx = _np3(aabb.get("max", [0, 0, 0]))
    return mn, mx


def detect_broken_connected(graph: Dict[str, Any], attach_eps: float, fp_eps: float) -> List[Dict[str, Any]]:
    """
    Returns list of broken connected relations, based on current node AABBs.
    """
    id2 = _id_to_node(graph)

    broken: List[Dict[str, Any]] = []
    for e in graph.get("edges", []):
        if str(e.get("type", "")) != "connected":
            continue

        a = int(e.get("a", -1))
        b = int(e.get("b", -1))
        if a not in id2 or b not in id2:
            continue

        na = id2[a]
        nb = id2[b]
        mnA, mxA = _node_aabb(na)
        mnB, mxB = _node_aabb(nb)

        ok, info = _is_connected(mnA, mxA, mnB, mxB, attach_eps=float(attach_eps))

        if not ok:
            broken.append({
                "type": "connected",
                "a": int(a),
                "b": int(b),
                "a_label": str(na.get("label", e.get("a_label", ""))),
                "b_label": str(nb.get("label", e.get("b_label", ""))),
                "was_attachment": e.get("attachment", ""),
                "broken": True,
                "metrics_now": info,
                "metrics_before": e.get("metrics", {}),  # what graph_building stored at creation time
            })

    return broken
