#!/usr/bin/env python3
"""
constraints_optimization/same_pair_loss.py

Same-pair size consistency loss (extent-wise percentage difference).

What this module does:
- Loads `same_pairs` from relations.json (records like {"a": "...", "b": "...", "confidence": ...}).
- Computes a size consistency loss between pairs of labels that should be the "same size".
- Your bbox labels are in raw format: "{base}_{x}" where {x} is a number.
  relations.json typically uses "{base}" (without the trailing _{x}).
- Therefore, we match a requested base label (e.g. "wheel_0") to raw bbox labels
  by searching raw labels that start with "{base}_" (and/or via a base->raw map).

Debugging:
- Default debug is False (quiet).
- If debug=True, prints candidate matches per pair and chosen best match.
"""

import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_same_pairs(relations_json_path: str) -> List[Dict[str, Any]]:
    """
    Load `same_pairs` from a relations.json file.

    Expected format:
      {"same_pairs": [{"a": "wheel_0", "b": "wheel_1", "confidence": 0.9, "evidence": "..."}]}
    """
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
        out.append(
            {
                "a": a,
                "b": b,
                "confidence": conf,
                "evidence": rec.get("evidence", ""),
            }
        )
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


_TRAILING_NUM_SUFFIX_ONCE = re.compile(r"^(.*)_(\d+)$")


def normalize_label_base(label: str) -> str:
    """
    Raw bbox labels are in format: {base}_{x}, where {x} is a number.
    Return base by stripping exactly one trailing _{number} if present.

    Examples:
      - "wheel_0_3" -> "wheel_0"
      - "chair"     -> "chair"
    """
    s = str(label)
    m = _TRAILING_NUM_SUFFIX_ONCE.match(s)
    if m:
        return m.group(1)
    return s


def build_base_to_raw_labels(label_to_extent: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
    """
    Build map: base_label -> [raw_bbox_labels]
    base_label is derived from raw label by stripping ONE trailing _{number}.
    """
    mp: Dict[str, List[str]] = {}
    for raw in label_to_extent.keys():
        base = normalize_label_base(raw)
        mp.setdefault(base, []).append(raw)
    return mp


def _extent_pct_diff(
    ea: np.ndarray, eb: np.ndarray, eps: float, reduce: str
) -> Tuple[float, np.ndarray]:
    """
    Compute per-axis relative extent difference:
      dxyz = |ea-eb| / max(ea, eb, eps)
    Reduce by mean or max to get scalar d in [0,1].
    """
    ea = np.maximum(np.asarray(ea, dtype=np.float64).reshape(3), 0.0)
    eb = np.maximum(np.asarray(eb, dtype=np.float64).reshape(3), 0.0)
    denom = np.maximum(np.maximum(ea, eb), float(eps))
    dxyz = np.abs(ea - eb) / denom
    if reduce == "max":
        d = float(np.max(dxyz))
    else:
        d = float(np.mean(dxyz))
    return float(np.clip(d, 0.0, 1.0)), dxyz


def same_pair_size_loss_0_1(
    *,
    same_pairs: List[Dict[str, Any]],
    label_to_extent: Dict[str, np.ndarray],
    eps: float = 1e-12,
    reduce: str = "mean",
    debug: bool = False,
) -> float:
    """
    Computes weighted average same-pair size mismatch in [0,1].

    Matching rule:
    - relations.json provides base labels: e.g. "wheel_0"
    - raw bbox labels are: e.g. "wheel_0_3", "wheel_0_7"
    - candidates are all raw labels that match the base:
        1) via base_map[base]
        2) fallback: raw labels starting with f"{base}_"
    - Among all candidate combinations, chooses the pair with MIN mismatch.

    Returns:
      L_same in [0,1]
    """
    if not same_pairs:
        if debug:
            print("[SAME_PAIR][DBG] no same_pairs -> L_same=0")
        return 0.0

    raw_labels = list(label_to_extent.keys())
    base_map = build_base_to_raw_labels(label_to_extent)

    if debug:
        base_keys = sorted(base_map.keys())
        print("[SAME_PAIR][DBG] where I look for bbox labels:")
        print("  - source: label_to_extent.keys() (built in optimizer from current boxes)")
        print(f"  - num raw bbox labels: {len(raw_labels)}")
        print(f"  - num base keys: {len(base_keys)}")
        for i, lab in enumerate(raw_labels[:10]):
            print(f"    raw[{i:02d}] = {lab}  -> base = {normalize_label_base(lab)}")
        if len(raw_labels) > 10:
            print("    ...")

    num = 0.0
    den = 0.0

    for k, rec in enumerate(same_pairs):
        a_base = str(rec["a"])
        b_base = str(rec["b"])
        w = float(rec.get("confidence", 1.0))

        # Primary lookup: base_map
        cand_a = list(base_map.get(a_base, []))
        cand_b = list(base_map.get(b_base, []))

        # Fallback: prefix scan "{base}_"
        if not cand_a:
            prefix = a_base + "_"
            cand_a = [lab for lab in raw_labels if lab.startswith(prefix)]
        if not cand_b:
            prefix = b_base + "_"
            cand_b = [lab for lab in raw_labels if lab.startswith(prefix)]

        if debug:
            print("\n" + "-" * 80)
            print(f"[SAME_PAIR][DBG] pair#{k:02d} request: a='{a_base}' b='{b_base}'  w={w:.3g}")
            print(f"[SAME_PAIR][DBG] found candidates for a='{a_base}': {'NO' if not cand_a else cand_a}")
            print(f"[SAME_PAIR][DBG] found candidates for b='{b_base}': {'NO' if not cand_b else cand_b}")

        if w <= 0.0:
            continue
        if not cand_a or not cand_b:
            if debug:
                print("[SAME_PAIR][DBG] -> skip (missing candidates)")
            continue

        best = None  # (d, la, lb, dxyz)
        for la in cand_a:
            ea = label_to_extent[la]
            for lb in cand_b:
                eb = label_to_extent[lb]
                d, dxyz = _extent_pct_diff(ea, eb, eps=eps, reduce=reduce)
                if (best is None) or (d < best[0]):
                    best = (d, la, lb, dxyz)

        if best is None:
            if debug:
                print("[SAME_PAIR][DBG] -> skip (no best)")
            continue

        d, la, lb, dxyz = best

        if debug:
            ea = np.asarray(label_to_extent[la], dtype=np.float64).reshape(3)
            eb = np.asarray(label_to_extent[lb], dtype=np.float64).reshape(3)
            print(f"[SAME_PAIR][DBG] chosen: la='{la}' lb='{lb}'")
            print(f"[SAME_PAIR][DBG] extent(la)={ea.tolist()}")
            print(f"[SAME_PAIR][DBG] extent(lb)={eb.tolist()}")
            print(f"[SAME_PAIR][DBG] dxyz={np.asarray(dxyz).tolist()} reduce={reduce} => d={d:.6g}")

        num += w * float(d)
        den += w

    if den <= 0.0:
        if debug:
            print("[SAME_PAIR][DBG] all pairs skipped -> L_same=0")
        return 0.0

    L = float(np.clip(num / den, 0.0, 1.0))
    if debug:
        print(f"[SAME_PAIR][DBG] L_same={L:.6g} (num={num:.6g}, den={den:.6g})")
    return L
