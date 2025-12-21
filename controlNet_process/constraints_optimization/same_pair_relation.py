#!/usr/bin/env python3
import os
import json
import re
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _deepcopy_jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x))

def _norm_label(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

_TRAILING_INDEX_RE = re.compile(r"(.*)_\d+$")

def _base_label(s: str) -> str:
    s = _norm_label(s)
    m = _TRAILING_INDEX_RE.match(s)
    return m.group(1) if m else s

def _load_primitives(primitives_json_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    raw = _load_json(primitives_json_path)
    if isinstance(raw, dict) and "primitives" in raw:
        return raw, raw["primitives"]
    if isinstance(raw, list):
        return {"primitives": raw}, raw
    raise ValueError(f"Unexpected primitives JSON format: {primitives_json_path}")


# -----------------------------------------------------------------------------
# Label matching: relation label -> primitive indices
# -----------------------------------------------------------------------------

def _build_label_index(primitives: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    exact = defaultdict(list)
    base = defaultdict(list)
    for i, p in enumerate(primitives):
        lab = _norm_label(p.get("label", "unknown"))
        exact[lab].append(i)
        base[_base_label(lab)].append(i)

    out = {}
    for k, v in exact.items():
        out[f"exact::{k}"] = v
    for k, v in base.items():
        out[f"base::{k}"] = v
    return out

def _match_relation_label(rel_label: str, label_index: Dict[str, List[int]]) -> Tuple[List[int], str]:
    r = _norm_label(rel_label)
    key = f"exact::{r}"
    if key in label_index:
        return label_index[key], "exact"

    rb = _base_label(r)
    keyb = f"base::{rb}"
    if keyb in label_index:
        return label_index[keyb], "base"

    return [], "none"


# -----------------------------------------------------------------------------
# Extent helpers (axis-permutation safe)
# -----------------------------------------------------------------------------

def _extent_perm(e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (perm, inv_perm) where:
      sorted_e = e[perm]
      original_e = sorted_e[inv_perm]
    """
    perm = np.argsort(e)  # ascending
    inv = np.empty_like(perm)
    inv[perm] = np.arange(3)
    return perm, inv


def _weighted_mean(sorted_exts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    sorted_exts: (M,3), weights: (M,)
    returns mean in sorted space.
    """
    if sorted_exts.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)
    w = weights.reshape(-1, 1)
    denom = max(1e-8, float(w.sum()))
    return (sorted_exts * w).sum(axis=0) / denom


def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * a + alpha * b


# -----------------------------------------------------------------------------
# Union-Find for same_pairs -> equivalence groups
# -----------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> List[List[str]]:
        roots = defaultdict(list)
        for x in list(self.parent.keys()):
            roots[self.find(x)].append(x)
        return [sorted(v) for v in roots.values()]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def apply_same_pair_relation(
    primitives_json_path: str,
    relations_json_path: str,
    out_dir: str,
    *,
    # Midpoint/mean behavior:
    use_pointcount_weights: bool = False,  # False => uniform => (A+B)/2 for pairs
    alpha: float = 1.0,                    # 1 => set to mean; <1 => partial move
    min_same_confidence: float = 0.0,
    min_group_members: int = 2,
) -> Dict[str, Any]:
    """
    Applies SAME constraints from relations.json same_pairs.

    Interpretation (what you asked):
      For same group, set all member sizes to the group's mean in a permutation-safe way.
      For a pair: A_new = B_new = (A+B)/2 (in sorted extent space).

    Does NOT change centers or rotations.

    Writes:
      out_dir/optimized_primitives.json
      out_dir/same_pair_report.json
    """
    os.makedirs(out_dir, exist_ok=True)

    prim_raw, primitives = _load_primitives(primitives_json_path)
    relations = _load_json(relations_json_path)
    same_pairs = relations.get("same_pairs", [])

    label_index = _build_label_index(primitives)

    # Build equivalence groups from same_pairs
    uf = UnionFind()
    used_pairs = []
    skipped_pairs = []

    for sp in same_pairs:
        a = sp.get("a", None)
        b = sp.get("b", None)
        conf = float(sp.get("confidence", 0.0))
        if a is None or b is None:
            continue
        if conf < min_same_confidence:
            skipped_pairs.append({"a": a, "b": b, "confidence": conf, "reason": f"confidence<{min_same_confidence}"})
            continue
        a_n = _norm_label(a)
        b_n = _norm_label(b)
        uf.union(a_n, b_n)
        used_pairs.append({"a": a_n, "b": b_n, "confidence": conf, "evidence": sp.get("evidence", "")})

    rel_groups = uf.groups()

    # output primitives copy
    new_primitives = [_deepcopy_jsonable(p) for p in primitives]

    report: Dict[str, Any] = {
        "inputs": {
            "primitives_json": os.path.abspath(primitives_json_path),
            "relations_json": os.path.abspath(relations_json_path),
        },
        "params": {
            "use_pointcount_weights": bool(use_pointcount_weights),
            "alpha": float(alpha),
            "min_same_confidence": float(min_same_confidence),
            "min_group_members": int(min_group_members),
        },
        "same_pairs_total": len(same_pairs),
        "same_pairs_used": len(used_pairs),
        "same_pairs_skipped": len(skipped_pairs),
        "same_pairs_used_list": used_pairs,
        "same_pairs_skipped_list": skipped_pairs,
        "groups": [],
        "unmatched_relation_labels": [],
        "changed_cluster_ids": [],
    }

    unmatched_rel_labels = set()
    changed_cids = set()

    for g in rel_groups:
        # map group relation labels -> primitive indices
        all_prim_idxs: List[int] = []
        match_details = []

        for rel_lab in g:
            idxs, mode = _match_relation_label(rel_lab, label_index)
            match_details.append({"relation_label": rel_lab, "match_mode": mode, "matched_primitive_indices": idxs})
            if not idxs:
                unmatched_rel_labels.add(rel_lab)
            all_prim_idxs.extend(idxs)

        # unique
        all_prim_idxs = sorted(list(dict.fromkeys(all_prim_idxs)))

        if len(all_prim_idxs) < min_group_members:
            report["groups"].append({
                "relation_group": g,
                "matched_primitive_count": len(all_prim_idxs),
                "skipped": True,
                "skip_reason": f"matched_primitive_count < {min_group_members}",
                "match_details": match_details,
            })
            continue

        # compute mean target in SORTED extent space
        M = len(all_prim_idxs)
        sorted_exts = np.zeros((M, 3), dtype=np.float64)
        weights = np.ones((M,), dtype=np.float64)
        inv_perms = []

        for k, pi in enumerate(all_prim_idxs):
            e = np.array(primitives[pi].get("parameters", {}).get("extent", [0, 0, 0]), dtype=np.float64)
            perm, inv = _extent_perm(e)
            sorted_exts[k] = e[perm]
            inv_perms.append(inv)
            if use_pointcount_weights:
                weights[k] = float(max(1, primitives[pi].get("point_count", 1)))

        target_sorted = _weighted_mean(sorted_exts, weights)

        members_out = []
        for k, pi in enumerate(all_prim_idxs):
            p_old = primitives[pi]
            p_new = new_primitives[pi]

            cid = int(p_old.get("cluster_id", -1))
            params_old = p_old.get("parameters", {})
            params_new = p_new.setdefault("parameters", {})

            e_old = np.array(params_old.get("extent", [0, 0, 0]), dtype=np.float64)
            perm_old, inv_old = _extent_perm(e_old)
            old_sorted = e_old[perm_old]

            new_sorted = _blend(old_sorted, target_sorted, alpha=alpha)
            e_new = new_sorted[inv_old]

            params_new["extent_before_opt"] = params_old.get("extent", [0, 0, 0])
            params_new["extent"] = e_new.tolist()

            changed_cids.add(cid)

            members_out.append({
                "cluster_id": cid,
                "primitive_label": p_old.get("label", "unknown"),
                "point_count": int(p_old.get("point_count", 0)),
                "extent_before": e_old.tolist(),
                "extent_after": e_new.tolist(),
                "extent_before_sorted": old_sorted.tolist(),
            })

        report["groups"].append({
            "relation_group": g,
            "matched_primitive_count": len(all_prim_idxs),
            "skipped": False,
            "target_sorted_extent_mean": target_sorted.tolist(),
            "match_details": match_details,
            "members": members_out,
        })

    report["unmatched_relation_labels"] = sorted(list(unmatched_rel_labels))
    report["changed_cluster_ids"] = sorted([x for x in changed_cids if x is not None])

    # save outputs
    out_primitives_path = os.path.join(out_dir, "optimized_primitives.json")
    out_report_path = os.path.join(out_dir, "same_pair_report.json")

    out_raw = dict(prim_raw) if isinstance(prim_raw, dict) else {"primitives": new_primitives}
    out_raw["primitives"] = new_primitives
    out_raw.setdefault("optimization", {})
    out_raw["optimization"].update({
        "method": "same_pairs_mean_extent",
        "alpha": float(alpha),
        "use_pointcount_weights": bool(use_pointcount_weights),
        "relations_json": os.path.abspath(relations_json_path),
        "source_primitives_json": os.path.abspath(primitives_json_path),
    })

    _save_json(out_primitives_path, out_raw)
    _save_json(out_report_path, report)

    return {
        "optimized_primitives_json": out_primitives_path,
        "report_json": out_report_path,
    }
