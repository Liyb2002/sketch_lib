#!/usr/bin/env python3
# AEP/attachment.py
#
# Step 1 (as you asked): identify attachment type (face / volume / point)
# and print them out for the target (and optionally the counts).
#
# This file keeps the same public API:
#   apply_attachments(constraints, edit, verbose=True) -> attach_res dict
#
# For now, it ONLY classifies + prints (no propagation yet).

from typing import Dict, Any, List, Optional


# ----------------------------
# Relation type inference (robust to older/newer formats)
# ----------------------------

def _infer_attachment_kind(e: Dict[str, Any]) -> str:
    """
    Returns one of: "volume", "face", "point"
    Priority:
      1) explicit keys if present
      2) volume metrics
      3) face keys
      4) fallback -> point
    """
    for k in ["kind", "attachment_kind", "relation_kind", "attachment_type", "relation_type"]:
        v = e.get(k, None)
        if isinstance(v, str):
            vv = v.lower()
            if "vol" in vv:
                return "volume"
            if "face" in vv:
                return "face"
            if "point" in vv or "anchor" in vv:
                return "point"

    # heuristics
    if any(x in e for x in ["overlap_volume", "vol_overlap", "overlap_box_local_min", "overlap_box_local_max"]):
        return "volume"
    if ("a_face" in e) and ("b_face" in e):
        return "face"
    return "point"


def _other_name_in_edge(e: Dict[str, Any], name: str) -> Optional[str]:
    a = e.get("a", None)
    b = e.get("b", None)
    if a == name and isinstance(b, str):
        return b
    if b == name and isinstance(a, str):
        return a
    return None


# ----------------------------
# Public API used by launcher
# ----------------------------

def apply_attachments(constraints: Dict[str, Any], edit: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Step 1 behavior:
      - Find all attachment edges that touch edit["target"]
      - Infer type: volume / face / point
      - Print them (and counts)
      - Return attach_res with a summary (no geometry changes yet)

    Returns:
      {
        "target": <str>,
        "applied": False,
        "changed_nodes": {},
        "summary": {
            "total_edges": int,
            "counts": {"volume":int,"face":int,"point":int,"unknown":int},
            "by_kind": {
                "volume": [ {...edge summary...}, ... ],
                "face":   [ ... ],
                "point":  [ ... ],
            }
        }
      }
    """
    attachments = constraints.get("attachments", []) or []

    target = edit.get("target", None)
    if not isinstance(target, str) or not target:
        raise ValueError("edit missing valid 'target'")

    # collect edges involving target
    target_edges: List[Dict[str, Any]] = []
    for e in attachments:
        if not isinstance(e, dict):
            continue
        if e.get("a") == target or e.get("b") == target:
            target_edges.append(e)

    counts = {"volume": 0, "face": 0, "point": 0, "unknown": 0}
    by_kind = {"volume": [], "face": [], "point": [], "unknown": []}

    def edge_summary(e: Dict[str, Any], kind: str) -> Dict[str, Any]:
        other = _other_name_in_edge(e, target)
        out = {
            "kind": kind,
            "a": e.get("a", None),
            "b": e.get("b", None),
            "other": other,
        }
        # include face ids if present
        if "a_face" in e or "b_face" in e:
            out["a_face"] = e.get("a_face", None)
            out["b_face"] = e.get("b_face", None)
        # include anchor if present (but keep small)
        if "anchor_world" in e:
            out["anchor_world"] = e.get("anchor_world", None)
        if "anchor_local" in e:
            out["anchor_local"] = e.get("anchor_local", None)
        # include volume metric hints if present
        for k in ["overlap_volume", "vol_overlap"]:
            if k in e:
                out[k] = e.get(k, None)
        return out

    for e in target_edges:
        kind = _infer_attachment_kind(e)
        if kind not in ("volume", "face", "point"):
            kind = "unknown"
        counts[kind] += 1
        by_kind[kind].append(edge_summary(e, kind))

    if verbose:
        print(f"[AEP][attach] target={target} attachment_edges={len(target_edges)}")
        print(f"[AEP][attach] counts: volume={counts['volume']} face={counts['face']} point={counts['point']} unknown={counts['unknown']}")

        # Print per-edge classification in a stable order (volume -> face -> point -> unknown)
        order = ["volume", "face", "point", "unknown"]
        for k in order:
            if len(by_kind[k]) == 0:
                continue
            print(f"\n[AEP][attach] ---- {k.upper()} edges ({len(by_kind[k])}) ----")
            for i, s in enumerate(by_kind[k]):
                # compact single-line print
                a = s.get("a")
                b = s.get("b")
                other = s.get("other")
                if k == "face":
                    print(f"[AEP][attach]   [{i}] {a} <-> {b} | other={other} | a_face={s.get('a_face')} b_face={s.get('b_face')}")
                elif k == "volume":
                    ov = s.get("overlap_volume", s.get("vol_overlap", None))
                    if ov is not None:
                        print(f"[AEP][attach]   [{i}] {a} <-> {b} | other={other} | overlap={ov}")
                    else:
                        print(f"[AEP][attach]   [{i}] {a} <-> {b} | other={other}")
                else:
                    aw = s.get("anchor_world", None)
                    if aw is not None:
                        print(f"[AEP][attach]   [{i}] {a} <-> {b} | other={other} | anchor_world={aw}")
                    else:
                        print(f"[AEP][attach]   [{i}] {a} <-> {b} | other={other}")

    return {
        "target": target,
        "applied": False,          # step 1 only
        "changed_nodes": {},       # no edits yet
        "summary": {
            "total_edges": int(len(target_edges)),
            "counts": counts,
            "by_kind": {
                "volume": by_kind["volume"],
                "face": by_kind["face"],
                "point": by_kind["point"],
                "unknown": by_kind["unknown"],
            },
        },
    }
