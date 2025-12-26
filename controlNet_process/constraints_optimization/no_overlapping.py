#!/usr/bin/env python3
"""
constraints_optimization/no_overlapping.py

Asymmetric shrink-only, value-aware bounding box optimization (balanced 0..1 losses),
with NONLINEAR value importance and NO shrink regularizer.

Backward-compat:
- accept w_shrink but ignore it (launcher may still pass it)

Goal:
- Minimize overlap between boxes using WORLD-AABB intersection volumes.
- Minimize value loss, where value(label, box) = sum of (heat^gamma) of points inside the box.
- Shrink-only: move faces inward independently (local min/max per axis).
- Rotation fixed. Center implied by optimized bounds and may move.

Balanced normalization (both in [0,1]):
- Overlap loss:
    L_ov = sum_{i<j} V_ij / max(eps, sum_{i<j} min(V_i, V_j))
  => L_ov in [0,1].

- Value loss:
    L_val = sum_i (value0_i - value_i) / max(eps, sum_i value0_i)
  where value uses heat^gamma.
  => L_val in [0,1].

Total core loss:
  L_core = w_overlap * L_ov + w_value * L_val

Printing:
- Every `print_every` iterations prints overlap/value/sum + overlap_pairs + raw inter_sum.

Exports:
- apply_no_overlapping_shrink_only(...)
- optimize_bounding_boxes(...)  # backward-compatible alias
"""

import os
import json
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------------------------------------------------------
# Heat decode (matches heat_map.py colormap)
# -----------------------------------------------------------------------------

def _heat_from_red_green_black(colors_0_1: np.ndarray) -> np.ndarray:
    """
    Reverse heat_map.py colormap:
      h<=0.5 : rgb=(0, 2h, 0)         => h=0.5*g
      h>=0.5 : rgb=(2h-1, 2-2h, 0)    => h=0.5+0.5*r
    """
    c = np.asarray(colors_0_1, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] < 2:
        return np.zeros((c.shape[0],), dtype=np.float32)
    r = c[:, 0]
    g = c[:, 1]
    h = np.where(r > 1e-6, 0.5 + 0.5 * r, 0.5 * g)
    return np.clip(h, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Geometry (LOCAL bounds -> WORLD AABB)
# -----------------------------------------------------------------------------

def _to_local(points_world: np.ndarray, center0_world: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Local frame anchored at original center0_world with axes R.
    """
    c = np.asarray(center0_world, dtype=np.float64).reshape(1, 3)
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return (np.asarray(points_world, dtype=np.float64) - c) @ Rm.T

def _local_center_from_bounds(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    bmin = np.asarray(bmin, dtype=np.float64).reshape(3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(3)
    return 0.5 * (bmin + bmax)

def _extent_from_bounds(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    bmin = np.asarray(bmin, dtype=np.float64).reshape(3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(3)
    return np.maximum(bmax - bmin, 0.0)

def _center_world_from_bounds(center0_world: np.ndarray, R: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    c0 = np.asarray(center0_world, dtype=np.float64).reshape(3)
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    lc = _local_center_from_bounds(bmin, bmax).reshape(3)
    return c0 + (Rm @ lc)

def _obb_corners_world_asym(center0_world: np.ndarray, R: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    c0 = np.asarray(center0_world, dtype=np.float64).reshape(3)
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    bmin = np.asarray(bmin, dtype=np.float64).reshape(3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(3)

    xs = [bmin[0], bmax[0]]
    ys = [bmin[1], bmax[1]]
    zs = [bmin[2], bmax[2]]

    corners_local = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    corners_world = (Rm @ corners_local.T).T + c0[None, :]
    return corners_world

def _obb_world_aabb_asym(center0_world: np.ndarray, R: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    corners = _obb_corners_world_asym(center0_world, R, bmin, bmax)
    return corners.min(axis=0), corners.max(axis=0)


# -----------------------------------------------------------------------------
# AABB overlap
# -----------------------------------------------------------------------------

def _box_volume(mn: np.ndarray, mx: np.ndarray) -> float:
    ext = np.asarray(mx, dtype=np.float64) - np.asarray(mn, dtype=np.float64)
    ext = np.maximum(ext, 0.0)
    return float(ext[0] * ext[1] * ext[2])

def _pairwise_overlap_volume(mn1: np.ndarray, mx1: np.ndarray, mn2: np.ndarray, mx2: np.ndarray) -> float:
    omax = np.minimum(mx1, mx2)
    omin = np.maximum(mn1, mn2)
    oext = np.maximum(omax - omin, 0.0)
    return float(oext[0] * oext[1] * oext[2])

def _compute_pairwise_overlaps(mins: np.ndarray, maxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)
    n = int(mins.shape[0])

    vols = np.array([_box_volume(mins[i], maxs[i]) for i in range(n)], dtype=np.float64)
    per_box = np.zeros((n,), dtype=np.float64)

    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            v = _pairwise_overlap_volume(mins[i], maxs[i], mins[j], maxs[j])
            if v > 0.0:
                pairs += 1
                total += v
                per_box[i] += v
                per_box[j] += v
    return vols, per_box, float(total), int(pairs)

def _overlap_loss_0_1(mins: np.ndarray, maxs: np.ndarray, eps: float = 1e-12) -> Tuple[float, float, float, int, np.ndarray, np.ndarray]:
    vols, per_box_overlap, inter_sum, overlap_pairs = _compute_pairwise_overlaps(mins, maxs)

    denom = 0.0
    n = int(mins.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            denom += float(min(vols[i], vols[j]))

    L = float(inter_sum) / max(float(eps), float(denom))
    L = float(np.clip(L, 0.0, 1.0))
    return L, float(inter_sum), float(denom), int(overlap_pairs), per_box_overlap, vols


# -----------------------------------------------------------------------------
# Value inside bounds (NONLINEAR heat^gamma)
# -----------------------------------------------------------------------------

def _load_heat_ply_points_and_heat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if o3d is None:
        raise RuntimeError("open3d required to read PLY. pip install open3d")
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts, np.zeros((0,), dtype=np.float32)
    heat = _heat_from_red_green_black(cols)
    return pts, heat

def _value_inside_bounds(local_pts0: np.ndarray, heat_pow: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    bmin = np.asarray(bmin, dtype=np.float64).reshape(1, 3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(1, 3)
    inside = np.all((local_pts0 >= bmin) & (local_pts0 <= bmax), axis=1)
    return float(np.sum(heat_pow[inside]))


# -----------------------------------------------------------------------------
# Optimizer
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
    heat_gamma: float = 2.0,
    print_every: int = 10,
    verbose: bool = True,
    w_shrink: float = 0.0,   # backward-compat: ignored
    **_ignored: Any,         # backward-compat: ignore any extra kwargs
) -> Dict[str, Any]:
    payload = _load_json(bbox_json)
    labels = payload.get("labels", [])
    if not labels:
        rep = {"ok": True, "note": "No labels in bbox_json", "iters": 0}
        _save_json(out_report_json, rep)
        _save_json(out_optimized_bbox_json, payload)
        return rep

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

        pts_w, heat = _load_heat_ply_points_and_heat(heat_ply)
        pts_l0 = _to_local(pts_w, center0_world=center0, R=Rm)

        heat_pow = np.power(np.clip(heat.astype(np.float64), 0.0, 1.0), gamma).astype(np.float64)

        half0 = 0.5 * extent0
        bmin0 = -half0
        bmax0 = +half0

        v0 = _value_inside_bounds(pts_l0, heat_pow, bmin0, bmax0)
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
            mn, mx = _obb_world_aabb_asym(it["center0"], it["R"], it["bmin"], it["bmax"])
            mins[k] = mn
            maxs[k] = mx
        return mins, maxs

    def value_loss_0_1() -> float:
        lost = 0.0
        for it in items:
            lost += float(it["value0"] - it["value"])
        return float(np.clip(lost / sum_value0, 0.0, 1.0))

    def objective(mins: np.ndarray, maxs: np.ndarray) -> Dict[str, Any]:
        ov_L, inter_sum, ov_denom, overlap_pairs, _, _ = _overlap_loss_0_1(mins, maxs)
        val_L = value_loss_0_1()

        ov_term = float(w_overlap) * float(ov_L)
        val_term = float(w_value) * float(val_L)
        core = ov_term + val_term

        return {
            "overlap_L": float(ov_L),
            "value_L": float(val_L),
            "overlap_term": float(ov_term),
            "value_term": float(val_term),
            "core_loss": float(core),
            "inter_sum": float(inter_sum),
            "overlap_denom": float(ov_denom),
            "overlap_pairs": int(overlap_pairs),
        }

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
            "core_loss": float(cur["core_loss"]),
            "overlap_pairs": int(cur["overlap_pairs"]),
            "inter_sum": float(cur["inter_sum"]),
        })

        if verbose and (print_every > 0) and (itn % int(print_every) == 0):
            print(
                f"[NO_OVERLAP][iter={itn:04d}] "
                f"ov={cur['overlap_term']:.6g} (L={cur['overlap_L']:.4f})  "
                f"val={cur['value_term']:.6g} (L={cur['value_L']:.4f})  "
                f"sum={cur['core_loss']:.6g}  "
                f"(pairs={cur['overlap_pairs']}, inter_sum={cur['inter_sum']:.6g}, step={cur_step:.4g})"
            )

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
                V_ij = _pairwise_overlap_volume(mins[i], maxs[i], mins[j], maxs[j])
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

                    old_value = it["value"]
                    it["bmin"] = bmin_new
                    it["bmax"] = bmax_new
                    it["value"] = _value_inside_bounds(it["pts_l0"], it["heat_pow"], bmin_new, bmax_new)

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
                it["value"] = _value_inside_bounds(it["pts_l0"], it["heat_pow"], bmin_new, bmax_new)

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
        it["value"] = _value_inside_bounds(it["pts_l0"], it["heat_pow"], bmin_best, bmax_best)

    mins_f, maxs_f = compute_world_aabbs()
    final = objective(mins_f, maxs_f)

    # write optimized bbox json
    out_payload = json.loads(json.dumps(payload))
    out_labels = out_payload.get("labels", [])

    def _write_back(rec: Dict[str, Any], it: Dict[str, Any]) -> None:
        c0 = np.asarray(it["center0"], dtype=np.float64)
        bmin = np.asarray(it["bmin"], dtype=np.float64)
        bmax = np.asarray(it["bmax"], dtype=np.float64)
        c1 = _center_world_from_bounds(c0, it["R"], bmin, bmax)
        e1 = _extent_from_bounds(bmin, bmax)

        rec["obb"]["center"] = c1.tolist()
        rec["obb"]["extent"] = e1.tolist()
        rec["opt_bounds_local"] = {"min": bmin.tolist(), "max": bmax.tolist()}
        rec["opt"] = {
            "heat_gamma": float(gamma),
            "value0": float(it["value0"]),
            "value": float(it["value"]),
            "value_loss": float(it["value0"] - it["value"]),
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
        "heat_gamma": float(gamma),
        "final": {
            "overlap_L": float(final["overlap_L"]),
            "value_L": float(final["value_L"]),
            "core_loss": float(final["core_loss"]),
            "overlap_pairs": int(final["overlap_pairs"]),
            "inter_sum": float(final["inter_sum"]),
        },
        "history": history,
        "note": "Backward compatible: accepts w_shrink but ignores it. No shrink penalty. Value uses heat^gamma.",
    }
    _save_json(out_report_json, report)
    return report


def optimize_bounding_boxes(**kwargs):
    return apply_no_overlapping_shrink_only(**kwargs)
