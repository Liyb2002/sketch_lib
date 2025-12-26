#!/usr/bin/env python3
"""
constraints_optimization/no_overlapping.py

Shrink-only, value-aware bounding box optimization.

Goal:
- Minimize AABB overlap between boxes (computed in WORLD coordinates).
- Minimize value loss, where value(label, box) = sum of heat values of points inside the OBB.
- Boxes can ONLY shrink (extent decreases); center and rotation fixed.

Critical normalization (so overlap competes with large value sums):
- Overlap objective uses a *dimensionless* normalized overlap:
    sum_{i<j}  V_ij / max(eps, min(V_i, V_j))
  where V_ij is raw AABB intersection volume and V_i is box i AABB volume.
- Value loss objective uses normalized value loss:
    sum_i (value0_i - value_i) / max(eps, value0_i)

Print requirement:
- Print "overlapping volume for each label":
  For each label i, we print sum_j V_ij (raw AABB inter_vol with all other boxes).

Exports:
- apply_no_overlapping_shrink_only(...)
- optimize_bounding_boxes(...)  # backward-compatible alias for launcher

Notes:
- This file assumes bbox_json is produced by compute_pca_bounding_boxes()
  and contains per-label:
    rec["obb"] = {center,R,extent}, and rec["heat_ply"] path.
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
    r = c[:, 0]
    g = c[:, 1]
    h = np.where(r > 1e-6, 0.5 + 0.5 * r, 0.5 * g)
    return np.clip(h, 0.0, 1.0)


# -----------------------------------------------------------------------------
# OBB -> world AABB (via corners)
# -----------------------------------------------------------------------------

def _obb_corners_world(center: np.ndarray, R: np.ndarray, extent: np.ndarray) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    e = np.asarray(extent, dtype=np.float64).reshape(3)
    h = 0.5 * e

    signs = np.array(
        [[-1, -1, -1],
         [-1, -1,  1],
         [-1,  1, -1],
         [-1,  1,  1],
         [ 1, -1, -1],
         [ 1, -1,  1],
         [ 1,  1, -1],
         [ 1,  1,  1]],
        dtype=np.float64,
    )
    corners_local = signs * h[None, :]
    corners_world = (Rm @ corners_local.T).T + c[None, :]
    return corners_world

def _obb_world_aabb(center: np.ndarray, R: np.ndarray, extent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    corners = _obb_corners_world(center, R, extent)
    mn = corners.min(axis=0)
    mx = corners.max(axis=0)
    return mn, mx


# -----------------------------------------------------------------------------
# AABB volume + overlap volume
# -----------------------------------------------------------------------------

def _box_volume(mn: np.ndarray, mx: np.ndarray) -> float:
    ext = np.asarray(mx, dtype=np.float64) - np.asarray(mn, dtype=np.float64)
    ext = np.maximum(ext, 0.0)
    return float(ext[0] * ext[1] * ext[2])

def _pairwise_overlap_volume(mn1: np.ndarray, mx1: np.ndarray, mn2: np.ndarray, mx2: np.ndarray) -> float:
    omax = np.minimum(mx1, mx2)
    omin = np.maximum(mn1, mn2)
    oext = omax - omin
    oext = np.maximum(oext, 0.0)
    return float(oext[0] * oext[1] * oext[2])


def _compute_pairwise_overlaps(
    mins: np.ndarray,
    maxs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Returns:
      vols: (N,) AABB volumes
      per_box_overlap: (N,) sum_j V_ij (raw), counting each pair into both endpoints
      total_overlap: sum_{i<j} V_ij
      overlap_pairs: count of pairs with V_ij > 0
    """
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


def _normalized_overlap_objective(mins: np.ndarray, maxs: np.ndarray, eps: float = 1e-12) -> float:
    """
    Dimensionless overlap objective:
      sum_{i<j} V_ij / max(eps, min(V_i, V_j))
    """
    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)
    n = int(mins.shape[0])
    vols = np.array([_box_volume(mins[i], maxs[i]) for i in range(n)], dtype=np.float64)

    tot = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            vij = _pairwise_overlap_volume(mins[i], maxs[i], mins[j], maxs[j])
            if vij <= 0.0:
                continue
            denom = max(float(eps), float(min(vols[i], vols[j])))
            tot += float(vij) / denom
    return float(tot)


# -----------------------------------------------------------------------------
# Value inside OBB (sum heat inside)
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

def _to_local(points_world: np.ndarray, center: np.ndarray, R: np.ndarray) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return (points_world - c) @ Rm.T

def _value_inside_halfext(local_pts: np.ndarray, heat: np.ndarray, half: np.ndarray) -> float:
    h = np.asarray(half, dtype=np.float64).reshape(1, 3)
    inside = np.all(np.abs(local_pts) <= h, axis=1)
    return float(np.sum(heat[inside]))


# -----------------------------------------------------------------------------
# Optimizer (shrink-only)
# -----------------------------------------------------------------------------

def apply_no_overlapping_shrink_only(
    *,
    bbox_json: str,
    out_optimized_bbox_json: str,
    out_report_json: str,
    max_iter: int = 400,
    step_frac: float = 0.08,
    step_decay: float = 0.5,
    min_extent_frac: float = 0.15,
    w_overlap: float = 3.0,
    w_value: float = 1.0,
    w_shrink: float = 0.05,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Shrink-only optimization.

    Prints:
      - initial per-label overlap volumes (raw)
      - final per-label overlap volumes (raw)
      - per-label extent/value changes at end
    """
    payload = _load_json(bbox_json)
    labels = payload.get("labels", [])
    if not labels:
        rep = {"ok": True, "note": "No labels in bbox_json", "iters": 0}
        _save_json(out_report_json, rep)
        _save_json(out_optimized_bbox_json, payload)
        return rep

    items: List[Dict[str, Any]] = []
    for rec in labels:
        obb = rec.get("obb", {})
        center = np.asarray(obb["center"], dtype=np.float64)
        Rm = np.asarray(obb["R"], dtype=np.float64)
        extent0 = np.asarray(obb["extent"], dtype=np.float64)

        heat_ply = rec.get("heat_ply", None)
        if heat_ply is None:
            raise ValueError("Missing 'heat_ply' in bbox labels.")

        pts_w, heat = _load_heat_ply_points_and_heat(heat_ply)
        pts_l = _to_local(pts_w, center=center, R=Rm)
        v0 = _value_inside_halfext(pts_l, heat, 0.5 * extent0)

        items.append({
            "label": rec.get("label", rec.get("sanitized", "unknown")),
            "rec": rec,
            "center": center,
            "R": Rm,
            "extent0": extent0.copy(),
            "extent_min": np.maximum(extent0 * float(min_extent_frac), 1e-9),
            "extent": extent0.copy(),
            "pts_l": pts_l,
            "heat": heat,
            "value0": float(v0),
            "value": float(v0),
        })

    def compute_world_aabbs() -> Tuple[np.ndarray, np.ndarray]:
        mins = np.zeros((len(items), 3), dtype=np.float64)
        maxs = np.zeros((len(items), 3), dtype=np.float64)
        for k, it in enumerate(items):
            mn, mx = _obb_world_aabb(it["center"], it["R"], it["extent"])
            mins[k] = mn
            maxs[k] = mx
        return mins, maxs

    def total_value_loss_norm(eps: float = 1e-12) -> float:
        s = 0.0
        for it in items:
            denom = max(float(eps), float(it["value0"]))
            s += float(it["value0"] - it["value"]) / denom
        return float(s)

    def shrink_penalty_norm(eps: float = 1e-12) -> float:
        s = 0.0
        for it in items:
            e0 = np.asarray(it["extent0"], dtype=np.float64)
            e = np.asarray(it["extent"], dtype=np.float64)
            s += float(np.sum((e0 - e) / np.maximum(e0, float(eps))))
        return float(s)

    def objective(mins: np.ndarray, maxs: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        overlap_norm = _normalized_overlap_objective(mins, maxs)
        vols, per_box_overlap, inter_sum, overlap_pairs = _compute_pairwise_overlaps(mins, maxs)
        J = (float(w_overlap) * float(overlap_norm)
             + float(w_value) * float(total_value_loss_norm())
             + float(w_shrink) * float(shrink_penalty_norm()))
        info = {
            "overlap_norm": float(overlap_norm),
            "inter_sum": float(inter_sum),
            "overlap_pairs": int(overlap_pairs),
            "per_box_overlap": per_box_overlap,
            "vols": vols,
        }
        return float(J), info

    # ---------------- initial report (per-label overlap volume) ----------------
    mins0, maxs0 = compute_world_aabbs()
    J0, info0 = objective(mins0, maxs0)

    if verbose:
        print("\n[NO_OVERLAP][INIT] overlap_pairs=", info0["overlap_pairs"], " inter_sum=", f"{info0['inter_sum']:.9g}")
        print("[NO_OVERLAP][INIT] per-label raw overlap volume (sum with others):")
        per = np.asarray(info0["per_box_overlap"], dtype=np.float64)
        order = np.argsort(-per)
        for idx in order:
            print(f"  {items[int(idx)]['label']}: overlap_sum={per[int(idx)]:.9g}")

    # ---------------- search (greedy coordinate shrink on worst pair) ----------------
    best_J = float(J0)
    best_state = [it["extent"].copy() for it in items]
    history: List[Dict[str, Any]] = []

    cur_step = float(step_frac)
    no_improve_rounds = 0

    for itn in range(int(max_iter)):
        mins, maxs = compute_world_aabbs()
        J, info = objective(mins, maxs)

        history.append({
            "iter": int(itn),
            "step_frac": float(cur_step),
            "objective": float(J),
            "overlap_norm": float(info["overlap_norm"]),
            "inter_sum": float(info["inter_sum"]),
            "overlap_pairs": int(info["overlap_pairs"]),
            "value_loss_norm": float(total_value_loss_norm()),
            "shrink_penalty_norm": float(shrink_penalty_norm()),
        })

        # Hard stop: NO overlap means raw inter_sum == 0
        if float(info["inter_sum"]) <= 0.0:
            if J < best_J:
                best_J = float(J)
                best_state = [it2["extent"].copy() for it2 in items]
            break

        # choose worst overlapping pair by raw intersection volume
        n = len(items)
        worst = None  # (V_ij, i, j)
        for i in range(n):
            for j in range(i + 1, n):
                V_ij = _pairwise_overlap_volume(mins[i], maxs[i], mins[j], maxs[j])
                if V_ij > 0.0:
                    if (worst is None) or (V_ij > worst[0]):
                        worst = (V_ij, i, j)

        if worst is None:
            if J < best_J:
                best_J = float(J)
                best_state = [it2["extent"].copy() for it2 in items]
            break

        _, i, j = worst

        # candidate: shrink one axis of either box in the worst pair
        candidates = []
        for box_id in (int(i), int(j)):
            it = items[box_id]
            e_old = it["extent"]

            for axis in (0, 1, 2):
                e_new = e_old.copy()
                e_new[axis] = max(it["extent_min"][axis], e_old[axis] * (1.0 - cur_step))
                if np.allclose(e_new, e_old, rtol=0, atol=1e-12):
                    continue

                old_extent = it["extent"].copy()
                old_value = it["value"]

                it["extent"] = e_new
                it["value"] = _value_inside_halfext(it["pts_l"], it["heat"], 0.5 * e_new)

                mins_t, maxs_t = compute_world_aabbs()
                J_t, _ = objective(mins_t, maxs_t)

                it["extent"] = old_extent
                it["value"] = old_value

                candidates.append((float(J_t), int(box_id), int(axis), e_new))

        if not candidates:
            no_improve_rounds += 1
        else:
            candidates.sort(key=lambda t: t[0])
            J_new, box_id, axis, e_new = candidates[0]

            if J_new + 1e-12 < J:
                it = items[box_id]
                it["extent"] = e_new
                it["value"] = _value_inside_halfext(it["pts_l"], it["heat"], 0.5 * e_new)

                if J_new + 1e-12 < best_J:
                    best_J = float(J_new)
                    best_state = [it2["extent"].copy() for it2 in items]

                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

        if no_improve_rounds >= 6:
            cur_step *= float(step_decay)
            no_improve_rounds = 0
            if cur_step < 0.005:
                break

    # restore best
    for it, e_best in zip(items, best_state):
        it["extent"] = e_best
        it["value"] = _value_inside_halfext(it["pts_l"], it["heat"], 0.5 * e_best)

    # final overlaps
    mins_f, maxs_f = compute_world_aabbs()
    J_f, info_f = objective(mins_f, maxs_f)

    # ---------------- print final per-label overlap volume ----------------
    if verbose:
        print("\n[NO_OVERLAP][FINAL] overlap_pairs=", info_f["overlap_pairs"], " inter_sum=", f"{info_f['inter_sum']:.9g}")
        print("[NO_OVERLAP][FINAL] per-label raw overlap volume (sum with others):")
        per = np.asarray(info_f["per_box_overlap"], dtype=np.float64)
        order = np.argsort(-per)
        for idx in order:
            print(f"  {items[int(idx)]['label']}: overlap_sum={per[int(idx)]:.9g}")

        print("\n[NO_OVERLAP] === Per-box changes (extent + value) ===")
        for k, it in enumerate(items):
            e0 = np.asarray(it["extent0"], dtype=np.float64)
            e1 = np.asarray(it["extent"], dtype=np.float64)
            de = e0 - e1
            rel = de / np.maximum(e0, 1e-12)

            v0 = float(it["value0"])
            v1 = float(it["value"])
            vloss = v0 - v1
            vloss_rel = vloss / max(v0, 1e-12)

            print(
                f"[NO_OVERLAP][BOX {k:02d}] {it['label']}\n"
                f"   extent0 = [{e0[0]:.6g}, {e0[1]:.6g}, {e0[2]:.6g}]\n"
                f"   extent  = [{e1[0]:.6g}, {e1[1]:.6g}, {e1[2]:.6g}]\n"
                f"   shrink  = [{de[0]:.6g}, {de[1]:.6g}, {de[2]:.6g}]  "
                f"(rel=[{rel[0]:.3%}, {rel[1]:.3%}, {rel[2]:.3%}])\n"
                f"   value0  = {v0:.6g}\n"
                f"   value   = {v1:.6g}\n"
                f"   v_loss  = {vloss:.6g}  (rel={vloss_rel:.3%})"
            )

    # write optimized bbox json
    out_payload = json.loads(json.dumps(payload))
    out_labels = out_payload.get("labels", [])

    if len(out_labels) == len(items):
        for rec, it in zip(out_labels, items):
            rec["obb"]["extent"] = it["extent"].tolist()
            rec["opt"] = {
                "value0": float(it["value0"]),
                "value": float(it["value"]),
                "value_loss": float(it["value0"] - it["value"]),
                "extent0": it["extent0"].tolist(),
                "extent": it["extent"].tolist(),
            }
    else:
        lab2it = {it["label"]: it for it in items}
        for rec in out_labels:
            lab = rec.get("label", rec.get("sanitized", "unknown"))
            if lab in lab2it:
                it = lab2it[lab]
                rec["obb"]["extent"] = it["extent"].tolist()
                rec["opt"] = {
                    "value0": float(it["value0"]),
                    "value": float(it["value"]),
                    "value_loss": float(it["value0"] - it["value"]),
                    "extent0": it["extent0"].tolist(),
                    "extent": it["extent"].tolist(),
                }

    _save_json(out_optimized_bbox_json, out_payload)

    report = {
        "ok": True,
        "in_bbox_json": os.path.abspath(bbox_json),
        "out_bbox_json": os.path.abspath(out_optimized_bbox_json),
        "labels": int(len(items)),
        "init": {
            "objective": float(J0),
            "overlap_norm": float(info0["overlap_norm"]),
            "inter_sum": float(info0["inter_sum"]),
            "overlap_pairs": int(info0["overlap_pairs"]),
        },
        "final": {
            "objective": float(J_f),
            "overlap_norm": float(info_f["overlap_norm"]),
            "inter_sum": float(info_f["inter_sum"]),
            "overlap_pairs": int(info_f["overlap_pairs"]),
            "value_loss_norm": float(total_value_loss_norm()),
            "shrink_penalty_norm": float(shrink_penalty_norm()),
        },
        "per_label": [
            {
                "label": it["label"],
                "value0": float(it["value0"]),
                "value": float(it["value"]),
                "value_loss": float(it["value0"] - it["value"]),
                "extent0": it["extent0"].tolist(),
                "extent": it["extent"].tolist(),
            }
            for it in items
        ],
        "history": history,
        "note": (
            "Overlap is WORLD AABB intersection volume. Objective uses normalized overlap "
            "sum(V_ij/min(V_i,V_j)) and normalized value loss."
        ),
    }
    _save_json(out_report_json, report)
    return report


# -----------------------------------------------------------------------------
# Backward-compatible alias
# -----------------------------------------------------------------------------

def optimize_bounding_boxes(**kwargs):
    return apply_no_overlapping_shrink_only(**kwargs)
