#!/usr/bin/env python3
import os
import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _load_primitives(primitives_json_path: str) -> List[Dict[str, Any]]:
    data = _load_json(primitives_json_path)
    if isinstance(data, dict) and "primitives" in data:
        return data["primitives"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected primitives JSON format: {primitives_json_path}")

def _wrap_primitives_like_input(primitives_json_path: str, new_primitives: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Preserve top-level metadata if the input is dict-form.
    """
    data = _load_json(primitives_json_path)
    if isinstance(data, dict) and "primitives" in data:
        out = dict(data)
        out["primitives"] = new_primitives
        return out
    return {"primitives": new_primitives}


# -----------------------------------------------------------------------------
# DSL parsing: equivalence_groups -> sets of instance type strings
# -----------------------------------------------------------------------------

def _norm_label(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

def _dsl_equivalence_type_groups(dsl_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Your DSL draft format:
      assembly.instances: [{id: "C01", type: "Front Wheel"}, ...]
      assembly.equivalence_groups: [{type_name: "...", member_ids:[...]}]

    We convert each equivalence group into:
      {
        "group_name": type_name,
        "member_instance_ids": [...],
        "member_types": ["front wheel", "rear wheel", ...]
      }

    Why: your equivalence groups often include different instance types (e.g. C01 Front Wheel, C07 Rear Wheel).
    """
    asm = dsl_json.get("assembly", {})
    instances = asm.get("instances", [])
    eq_groups = asm.get("equivalence_groups", [])

    id_to_type = {}
    for inst in instances:
        iid = inst.get("id", None)
        itype = inst.get("type", None)
        if iid is None or itype is None:
            continue
        id_to_type[str(iid)] = _norm_label(str(itype))

    out = []
    for g in eq_groups:
        name = str(g.get("type_name", "unknown"))
        member_ids = [str(x) for x in g.get("member_ids", [])]
        member_types = []
        for mid in member_ids:
            if mid in id_to_type:
                member_types.append(id_to_type[mid])
        # unique, stable
        member_types = sorted(list(dict.fromkeys(member_types)))
        out.append({
            "group_name": name,
            "member_instance_ids": member_ids,
            "member_types": member_types,
        })
    return out


# -----------------------------------------------------------------------------
# Core extent-tying logic
# -----------------------------------------------------------------------------

@dataclass
class ExtentPerm:
    perm: np.ndarray        # argsort indices, shape (3,)
    inv_perm: np.ndarray    # inverse permutation, shape (3,)

def _extent_perm(e: np.ndarray) -> ExtentPerm:
    perm = np.argsort(e)  # ascending
    inv = np.empty_like(perm)
    inv[perm] = np.arange(3)
    return ExtentPerm(perm=perm, inv_perm=inv)

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    values: (M,), weights: (M,)
    """
    if values.size == 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * float(np.sum(w))
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    idx = max(0, min(idx, v.size - 1))
    return float(v[idx])

def _group_canonical_sorted_extent(
    sorted_extents: np.ndarray,  # (M,3)
    weights: np.ndarray,         # (M,)
    use_median: bool = True,
) -> np.ndarray:
    """
    Compute a canonical 3-vector in "sorted extent space" (ascending).
    Median is more robust than mean.
    """
    if sorted_extents.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)

    if use_median:
        out = []
        for k in range(3):
            out.append(_weighted_median(sorted_extents[:, k], weights))
        return np.array(out, dtype=np.float64)

    # weighted mean
    w = weights.reshape(-1, 1)
    return (sorted_extents * w).sum(axis=0) / max(1e-8, float(w.sum()))

def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    # alpha=1 -> b, alpha=0 -> a
    return (1.0 - alpha) * a + alpha * b


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def optimize_bbx_sizes_from_equivalence(
    primitives_json_path: str,
    dsl_json_path: str,
    out_dir: str,
    *,
    ply_path: Optional[str] = None,
    alpha: float = 1.0,
    min_group_members: int = 2,
    use_pointcount_weights: bool = True,
    canonical_use_median: bool = True,
) -> Dict[str, Any]:
    """
    Optimizes ONLY extents (size) based on DSL equivalence groups.
    Keeps centers + rotations unchanged.

    Matching rule:
      - DSL equivalence group -> set of member *types* (strings)
      - A primitive belongs to a group if its primitive["label"] matches one of those types (case-insensitive normalized)

    Because you said:
      - DSL relations are correct if they reference real components
      - some DSL components don't exist -> we just won't match them

    Saves:
      - optimized_primitives.json
      - optimized_report.json
    """
    os.makedirs(out_dir, exist_ok=True)

    primitives = _load_primitives(primitives_json_path)
    dsl = _load_json(dsl_json_path)

    eq_type_groups = _dsl_equivalence_type_groups(dsl)

    # index primitives by normalized label
    label_to_prim_idxs: Dict[str, List[int]] = {}
    for i, p in enumerate(primitives):
        lab = _norm_label(p.get("label", "unknown"))
        label_to_prim_idxs.setdefault(lab, []).append(i)

    # will modify copies
    new_primitives = [json.loads(json.dumps(p)) for p in primitives]  # deep copy via json

    report = {
        "inputs": {
            "primitives_json": os.path.abspath(primitives_json_path),
            "dsl_json": os.path.abspath(dsl_json_path),
            "ply_path": os.path.abspath(ply_path) if ply_path else None,
        },
        "params": {
            "alpha": float(alpha),
            "min_group_members": int(min_group_members),
            "use_pointcount_weights": bool(use_pointcount_weights),
            "canonical_use_median": bool(canonical_use_median),
        },
        "groups": [],
        "unmatched_group_types": [],
        "changed_clusters": [],
    }

    changed = set()

    for g in eq_type_groups:
        member_types = g.get("member_types", [])
        # collect all primitives whose label matches any member type
        prim_idxs = []
        matched_types = []
        for t in member_types:
            if t in label_to_prim_idxs:
                prim_idxs.extend(label_to_prim_idxs[t])
                matched_types.append(t)

        prim_idxs = sorted(list(dict.fromkeys(prim_idxs)))  # unique, stable

        if len(prim_idxs) < min_group_members:
            # This group doesn't exist in current clusters (or only 1 instance exists) -> ignore for optimization
            missing = [t for t in member_types if t not in label_to_prim_idxs]
            report["groups"].append({
                "group_name": g.get("group_name", "unknown"),
                "member_types": member_types,
                "matched_types": matched_types,
                "matched_primitive_count": len(prim_idxs),
                "skipped": True,
                "skip_reason": f"matched_primitive_count < {min_group_members}",
                "missing_types": missing,
            })
            if missing:
                report["unmatched_group_types"].extend(missing)
            continue

        # gather extents (and their sort perms)
        M = len(prim_idxs)
        sorted_exts = np.zeros((M, 3), dtype=np.float64)
        inv_perms = []
        weights = np.ones((M,), dtype=np.float64)

        for k, pi in enumerate(prim_idxs):
            params = primitives[pi].get("parameters", {})
            e = np.array(params.get("extent", [0, 0, 0]), dtype=np.float64)

            ep = _extent_perm(e)
            sorted_exts[k] = e[ep.perm]
            inv_perms.append(ep.inv_perm)

            if use_pointcount_weights:
                weights[k] = float(max(1, primitives[pi].get("point_count", 1)))

        canonical_sorted = _group_canonical_sorted_extent(
            sorted_extents=sorted_exts,
            weights=weights,
            use_median=canonical_use_median,
        )

        # apply canonical extents back to each primitive (in its original axis order),
        # optionally blending with original via alpha
        per_member = []
        for k, pi in enumerate(prim_idxs):
            p_old = primitives[pi]
            p_new = new_primitives[pi]
            params_old = p_old.get("parameters", {})
            params_new = p_new.setdefault("parameters", {})

            e_old = np.array(params_old.get("extent", [0, 0, 0]), dtype=np.float64)
            inv_perm = inv_perms[k]

            old_sorted = e_old[_extent_perm(e_old).perm]
            new_sorted = _blend(old_sorted, canonical_sorted, alpha=alpha)

            # map back to original axis order
            e_new = new_sorted[inv_perm]

            # record
            params_new["extent_before_opt"] = params_old.get("extent", [0, 0, 0])
            params_new["extent"] = e_new.tolist()

            changed.add(int(p_old.get("cluster_id", -1)))
            per_member.append({
                "cluster_id": int(p_old.get("cluster_id", -1)),
                "label": p_old.get("label", "unknown"),
                "point_count": int(p_old.get("point_count", 0)),
                "extent_before": e_old.tolist(),
                "extent_after": e_new.tolist(),
                "extent_before_sorted": old_sorted.tolist(),
            })

        report["groups"].append({
            "group_name": g.get("group_name", "unknown"),
            "member_types": member_types,
            "matched_types": matched_types,
            "matched_primitive_count": len(prim_idxs),
            "skipped": False,
            "canonical_sorted_extent": canonical_sorted.tolist(),
            "members": per_member,
        })

    report["unmatched_group_types"] = sorted(list(dict.fromkeys(report["unmatched_group_types"])))
    report["changed_clusters"] = sorted(list(changed))

    # save
    out_primitives_path = os.path.join(out_dir, "optimized_primitives.json")
    out_report_path = os.path.join(out_dir, "optimized_report.json")

    wrapped = _wrap_primitives_like_input(primitives_json_path, new_primitives)

    # attach metadata about optimization
    if isinstance(wrapped, dict):
        wrapped.setdefault("optimization", {})
        wrapped["optimization"].update({
            "method": "extent_tying_from_dsl_equivalence",
            "alpha": float(alpha),
            "dsl_json": os.path.abspath(dsl_json_path),
            "source_primitives_json": os.path.abspath(primitives_json_path),
        })

    _save_json(out_primitives_path, wrapped)
    _save_json(out_report_path, report)

    return {
        "optimized_primitives_json": out_primitives_path,
        "optimized_report_json": out_report_path,
    }
