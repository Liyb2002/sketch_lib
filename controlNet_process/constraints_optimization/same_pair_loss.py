#!/usr/bin/env python3
"""
constraints_optimization/same_pair_loss.py

Same-pair size consistency loss (percentage-based).

Reads "same_pairs" from relations.json and computes a [0,1] penalty that grows
when same-pair boxes have different sizes.

Loss definition (extent-wise percentage difference, normalized):
  For each pair (a,b), extents ea, eb (3,):
    d_k = |ea_k - eb_k| / max(eps, max(ea_k, eb_k))   in [0,1]
    d   = mean_k(d_k)   (or use max_k for stronger)
  L_same = weighted average(d, weight=confidence)

If a label in same_pairs doesn't exist in current boxes, that pair is skipped.
"""

import json
from typing import Any, Dict, List

import numpy as np


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_same_pairs(relations_json_path: str) -> List[Dict[str, Any]]:
    try:
        data = _load_json(relations_json_path)
    except Exception:
        return []

    sp = data.get("same_pairs", [])
    if not isinstance(sp, list):
        return []

    out: List[Dict[str, Any]] = []
    for rec in sp:
        if not isinstance(rec, dict):
            continue
        a = rec.get("a", None)
        b = rec.get("b", None)
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        conf = rec.get("confidence", 1.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 1.0
        conf = float(np.clip(conf, 0.0, 1.0))
        out.append({
            "a": a,
            "b": b,
            "confidence": conf,
            "evidence": rec.get("evidence", ""),
        })
    return out


def print_same_pairs(same_pairs: List[Dict[str, Any]]) -> None:
    if not same_pairs:
        print("[SAME_PAIR] same_pairs: (none)")
        return
    print("[SAME_PAIR] same_pairs:")
    for k, rec in enumerate(same_pairs):
        a = rec.get("a", "?")
        b = rec.get("b", "?")
        c = rec.get("confidence", 1.0)
        ev = rec.get("evidence", "")
        print(f"  - {k:02d}: {a} <-> {b}  (conf={c:.3g})  {ev}")


def same_pair_size_loss_0_1(
    *,
    same_pairs: List[Dict[str, Any]],
    label_to_extent: Dict[str, np.ndarray],
    eps: float = 1e-12,
    reduce: str = "mean",  # "mean" or "max"
) -> float:
    """
    label_to_extent: {label: extent(3,)} (world units)
    Returns L_same in [0,1] using extent-wise percentage difference.

    reduce:
      - "mean": average of x/y/z percentage diffs (default)
      - "max": max of x/y/z percentage diffs (stronger)
    """
    if not same_pairs:
        return 0.0

    num = 0.0
    den = 0.0

    for rec in same_pairs:
        a = rec["a"]
        b = rec["b"]
        w = float(rec.get("confidence", 1.0))
        if w <= 0.0:
            continue
        if a not in label_to_extent or b not in label_to_extent:
            continue

        ea = np.maximum(np.asarray(label_to_extent[a], dtype=np.float64).reshape(3), 0.0)
        eb = np.maximum(np.asarray(label_to_extent[b], dtype=np.float64).reshape(3), 0.0)

        denom = np.maximum(np.maximum(ea, eb), float(eps))  # per-axis
        dxyz = np.abs(ea - eb) / denom                      # per-axis in [0,1] (bounded)

        if reduce == "max":
            d = float(np.max(dxyz))
        else:
            d = float(np.mean(dxyz))

        d = float(np.clip(d, 0.0, 1.0))

        num += w * d
        den += w

    if den <= 0.0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))
