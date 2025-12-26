#!/usr/bin/env python3
"""
constraints_optimization/optimizer.py

(Updated) Adds SAME-PAIR size loss from constraints_optimization/same_pair_loss.py

New term:
  L_same in [0,1] penalizes different volumes for same_pairs (e.g., wheel_0 vs wheel_1)

Total:
  L_core = w_overlap*L_ov + w_value*L_val + w_same*L_same

Backward-compat:
- accept w_shrink but ignore it
- accept extra kwargs and ignore them
"""

import os
import json
from typing import Any, Dict, List, Tuple

import numpy as np

from constraints_optimization.no_overlap_loss import (
    load_json,
    load_heat_ply_points_and_heat,
    to_local,
    value_inside_bounds,
    obb_world_aabb_asym,
    center_world_from_bounds,
    extent_from_bounds,
    objective_from_world_aabbs,
    pairwise_overlap_volume,
)

from constraints_optimization.same_pair_loss import (
    load_same_pairs,
    print_same_pairs,
    same_pair_size_loss_0_1,
)


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _infer_relations_json_from_bbox_json(bbox_json: str) -> str:
    """
    bbox_json example:
      .../sketch/dsl_optimize/optimize_iteration/iter_000/heat_map/pca_bboxes/pca_bboxes.json
    relations.json is expected at:
      .../sketch/dsl_optimize/relations.json
    """
    p = os.path.abspath(bbox_json)
    marker = os.sep + "sketch" + os.sep
    idx = p.rfind(marker)
    if idx < 0:
        # best-effort fallback: assume cwd has sketch/dsl_optimize/relations.json
        return os.path.join(os.getcwd(), "sketch", "dsl_optimize", "relations.json")
    root = p[: idx + len(marker)]  # ends with ".../sketch/"
    return os.path.join(root, "dsl_optimize", "relations.json")


# -----------------------------------------------------------------------------
# Optimizer entry
# -----------------------------------------------------------------------------

def apply_no_overlapping_shrink_only(
    *,
    bbox_json: str,
    out_optimized_bbox_json: str,
    out_report_json: str,
    max_iter: int = 600,
    step_frac: float = 0.08,
    step_decay: float = 0.5,
    min_extent_frac: float = 0.15,
    w_overlap: float = 1.0,
    w_value: float = 1.0,
    w_same: float = 0.0,     # NEW: same-pair size consistency weight
    heat_gamma: float = 2.0,
    print_every: int = 10,
    verbose: bool = True,
    w_shrink: float = 0.0,   # backward-compat: ignored
    **_ignored: Any,         # backward-compat: ignore any extra kwargs
) -> Dict[str, Any]:
    payload = load_json(bbox_json)
    labels = payload.get("labels", [])
    if not labels:
        rep = {"ok": True, "note": "No labels in bbox_json", "iters": 0}
        _save_json(out_report_json, rep)
        _save_json(out_optimized_bbox_json, payload)
        return rep

    # ---- load same_pairs and print them once ----
    relations_json = _infer_relations_json_from_bbox_json(bbox_json)
    same_pairs = load_same_pairs(relations_json)
    if verbose:
        print(f"[SAME_PAIR] relations_json: {relations_json}")
        print_same_pairs(same_pairs)

    gamma = float(heat_gamma)
    if gamma <= 0.0 or not np.isfinite(gamma):
        gamma = 2.0

    items: List[Dict[str, Any]] = []
    for rec in labels:
        obb = rec.get("obb", {})
        center0 = np.asarray(obb["center"], dtype=np.float64).reshape(3)
        Rm = np.asarray(obb["R"], dtype=np.float64).reshape(3, 3)
        extent0 = np.asarray(obb["extent"], dtype=np.float64).reshape(3)

        heat_ply = rec.get("heat_ply", None)
        if heat_ply is None:
            raise ValueError("Missing 'heat_ply' in bbox labels.")

        pts_w, heat = load_heat_ply_points_and_heat(heat_ply)
        pts_l0 = to_local(pts_w, center0_world=center0, R=Rm)

        heat_pow = np.power(np.clip(heat.astype(np.float64), 0.0, 1.0), gamma).astype(np.float64)

        half0 = 0.5 * extent0
        bmin0 = -half0
        bmax0 = +half0

        v0 = value_inside_bounds(pts_l0, heat_pow, bmin0, bmax0)
        extent_min = np.maximum(extent0 * float(min_extent_frac), 1e-9)

        items.append({
            "label": rec.get("label", rec.get("sanitized", "unknown")),
            "rec": rec,
            "center0": center0,
            "R": Rm,
            "extent0": extent0.copy(),
            "extent_min": extent_min,
            "bmin0": bmin0.copy(),
            "bmax0": bmax0.copy(),
            "bmin": bmin0.copy(),
            "bmax": bmax0.copy(),
            "pts_l0": pts_l0,
            "heat_pow": heat_pow,
            "value0": float(v0),
            "value": float(v0),
        })

    sum_value0 = float(sum(it["value0"] for it in items))
    sum_value0 = max(1e-12, sum_value0)

    def compute_world_aabbs() -> Tuple[np.ndarray, np.ndarray]:
        mins = np.zeros((len(items), 3), dtype=np.float64)
        maxs = np.zeros((len(items), 3), dtype=np.float64)
        for k, it in enumerate(items):
            mn, mx = obb_world_aabb_asym(it["center0"], it["R"], it["bmin"], it["bmax"])
            mins[k] = mn
            maxs[k] = mx
        return mins, maxs

    def _label_to_extent_current() -> Dict[str, np.ndarray]:
        m: Dict[str, np.ndarray] = {}
        for it in items:
            m[it["label"]] = extent_from_bounds(it["bmin"], it["bmax"])
        return m

    def objective(mins: np.ndarray, maxs: np.ndarray) -> Dict[str, Any]:
        base = objective_from_world_aabbs(
            mins=mins,
            maxs=maxs,
            items=items,
            sum_value0=sum_value0,
            w_overlap=w_overlap,
            w_value=w_value,
        )

        L_same = same_pair_size_loss_0_1(
            same_pairs=same_pairs,
            label_to_extent=_label_to_extent_current(),
        )
        same_term = float(w_same) * float(L_same)

        core = float(base["core_loss"]) + same_term

        base.update({
            "same_L": float(L_same),
            "same_term": float(same_term),
            "core_loss": float(core),  # overwrite with new total
        })
        return base

    def _can_apply_bounds(it: Dict[str, Any], bmin_new: np.ndarray, bmax_new: np.ndarray) -> bool:
        bmin_new = np.asarray(bmin_new, dtype=np.float64).reshape(3)
        bmax_new = np.asarray(bmax_new, dtype=np.float64).reshape(3)
        ext = bmax_new - bmin_new
        if np.any(ext <= 0.0):
            return False
        if np.any(ext < np.asarray(it["extent_min"], dtype=np.float64)):
            return False
        return True

    # init
    mins0, maxs0 = compute_world_aabbs()
    cur = objective(mins0, maxs0)

    best_core = float(cur["core_loss"])
    best_state = [(it["bmin"].copy(), it["bmax"].copy()) for it in items]

    history: List[Dict[str, Any]] = []
    cur_step = float(step_frac)
    no_improve_rounds = 0

    for itn in range(int(max_iter)):
        mins, maxs = compute_world_aabbs()
        cur = objective(mins, maxs)

        history.append({
            "iter": int(itn),
            "step_frac": float(cur_step),
            "overlap_L": float(cur["overlap_L"]),
            "value_L": float(cur["value_L"]),
            "same_L": float(cur.get("same_L", 0.0)),
            "core_loss": float(cur["core_loss"]),
            "overlap_pairs": int(cur["overlap_pairs"]),
            "inter_sum": float(cur["inter_sum"]),
        })

        if verbose and (print_every > 0) and (itn % int(print_every) == 0):
            print(
                f"[NO_OVERLAP][iter={itn:04d}] "
                f"ov={cur['overlap_term']:.6g} (L={cur['overlap_L']:.4f})  "
                f"val={cur['value_term']:.6g} (L={cur['value_L']:.4f})  "
                f"same={cur.get('same_term', 0.0):.6g} (L={cur.get('same_L', 0.0):.4f})  "
                f"sum={cur['core_loss']:.6g}  "
                f"(pairs={cur['overlap_pairs']}, inter_sum={cur['inter_sum']:.6g}, step={cur_step:.4g})"
            )

        # if no overlap left, stop
        if float(cur["inter_sum"]) <= 0.0:
            if float(cur["core_loss"]) < best_core:
                best_core = float(cur["core_loss"])
                best_state = [(it2["bmin"].copy(), it2["bmax"].copy()) for it2 in items]
            break

        # worst overlapping pair by raw inter volume
        n = len(items)
        worst = None
        for i in range(n):
            for j in range(i + 1, n):
                V_ij = pairwise_overlap_volume(mins[i], maxs[i], mins[j], maxs[j])
                if V_ij > 0.0 and ((worst is None) or (V_ij > worst[0])):
                    worst = (V_ij, i, j)

        if worst is None:
            if float(cur["core_loss"]) < best_core:
                best_core = float(cur["core_loss"])
                best_state = [(it2["bmin"].copy(), it2["bmax"].copy()) for it2 in items]
            break

        _, i_w, j_w = worst

        candidates = []
        for box_id in (int(i_w), int(j_w)):
            it = items[box_id]
            bmin_old = it["bmin"].copy()
            bmax_old = it["bmax"].copy()
            ext_old = bmax_old - bmin_old

            for axis in (0, 1, 2):
                delta = float(cur_step) * float(ext_old[axis])
                if not np.isfinite(delta) or delta <= 0.0:
                    continue

                for side in ("min", "max"):
                    bmin_new = bmin_old.copy()
                    bmax_new = bmax_old.copy()
                    if side == "min":
                        bmin_new[axis] = bmin_new[axis] + delta
                    else:
                        bmax_new[axis] = bmax_new[axis] - delta

                    if not _can_apply_bounds(it, bmin_new, bmax_new):
                        continue

                    # try
                    old_value = it["value"]
                    it["bmin"] = bmin_new
                    it["bmax"] = bmax_new
                    it["value"] = value_inside_bounds(it["pts_l0"], it["heat_pow"], bmin_new, bmax_new)

                    mins_t, maxs_t = compute_world_aabbs()
                    cand = objective(mins_t, maxs_t)

                    # rollback
                    it["bmin"] = bmin_old
                    it["bmax"] = bmax_old
                    it["value"] = old_value

                    candidates.append((float(cand["core_loss"]), int(box_id), int(axis), side, bmin_new, bmax_new))

        if not candidates:
            no_improve_rounds += 1
        else:
            candidates.sort(key=lambda t: t[0])
            core_new, box_id, axis, side, bmin_new, bmax_new = candidates[0]

            if core_new + 1e-12 < float(cur["core_loss"]):
                it = items[box_id]
                it["bmin"] = bmin_new
                it["bmax"] = bmax_new
                it["value"] = value_inside_bounds(it["pts_l0"], it["heat_pow"], bmin_new, bmax_new)

                mins_b, maxs_b = compute_world_aabbs()
                cur_b = objective(mins_b, maxs_b)
                if float(cur_b["core_loss"]) < best_core:
                    best_core = float(cur_b["core_loss"])
                    best_state = [(it2["bmin"].copy(), it2["bmax"].copy()) for it2 in items]

                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

        if no_improve_rounds >= 6:
            cur_step *= float(step_decay)
            no_improve_rounds = 0
            if cur_step < 0.005:
                break

    # restore best
    for it, (bmin_best, bmax_best) in zip(items, best_state):
        it["bmin"] = bmin_best
        it["bmax"] = bmax_best
        it["value"] = value_inside_bounds(it["pts_l0"], it["heat_pow"], bmin_best, bmax_best)

    mins_f, maxs_f = compute_world_aabbs()
    final = objective(mins_f, maxs_f)

    # write optimized bbox json
    out_payload = json.loads(json.dumps(payload))
    out_labels = out_payload.get("labels", [])

    def _write_back(rec: Dict[str, Any], it: Dict[str, Any]) -> None:
        c0 = np.asarray(it["center0"], dtype=np.float64)
        bmin = np.asarray(it["bmin"], dtype=np.float64)
        bmax = np.asarray(it["bmax"], dtype=np.float64)
        c1 = center_world_from_bounds(c0, it["R"], bmin, bmax)
        e1 = extent_from_bounds(bmin, bmax)

        rec["obb"]["center"] = c1.tolist()
        rec["obb"]["extent"] = e1.tolist()
        rec["opt_bounds_local"] = {"min": bmin.tolist(), "max": bmax.tolist()}
        rec["opt"] = {
            "heat_gamma": float(gamma),
            "value0": float(it["value0"]),
            "value": float(it["value"]),
            "value_loss": float(it["value0"] - it["value"]),
            "same_pairs": len(same_pairs),
            "center0": it["center0"].tolist(),
            "center": c1.tolist(),
            "extent0": it["extent0"].tolist(),
            "extent": e1.tolist(),
        }

    if len(out_labels) == len(items):
        for rec, it in zip(out_labels, items):
            _write_back(rec, it)
    else:
        lab2it = {it["label"]: it for it in items}
        for rec in out_labels:
            lab = rec.get("label", rec.get("sanitized", "unknown"))
            if lab in lab2it:
                _write_back(rec, lab2it[lab])

    _save_json(out_optimized_bbox_json, out_payload)

    report = {
        "ok": True,
        "in_bbox_json": os.path.abspath(bbox_json),
        "out_bbox_json": os.path.abspath(out_optimized_bbox_json),
        "labels": int(len(items)),
        "relations_json": os.path.abspath(relations_json),
        "same_pairs": same_pairs,
        "heat_gamma": float(gamma),
        "weights": {
            "w_overlap": float(w_overlap),
            "w_value": float(w_value),
            "w_same": float(w_same),
        },
        "final": {
            "overlap_L": float(final["overlap_L"]),
            "value_L": float(final["value_L"]),
            "same_L": float(final.get("same_L", 0.0)),
            "core_loss": float(final["core_loss"]),
            "overlap_pairs": int(final["overlap_pairs"]),
            "inter_sum": float(final["inter_sum"]),
        },
        "history": history,
        "note": "Backward compatible: accepts w_shrink but ignores it. Adds same-pair size penalty via relations.json.",
    }
    _save_json(out_report_json, report)
    return report


def optimize_bounding_boxes(**kwargs):
    # backward-compatible alias
    return apply_no_overlapping_shrink_only(**kwargs)
