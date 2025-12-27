#!/usr/bin/env python3
"""
constraints_optimization/optimizer.py

Gradient-based (Adam) shrink-only optimizer over WORLD AABB boxes.

What it does now (per your spec):
1) Optimization method: gradient descent (Adam) on all boxes at once.
2) Directly optimizes WORLD AABB mins/maxs via (mid, half) parameters; boxes remain axis-aligned.
3) Updates all boxes simultaneously every gradient step (no explicit per-box responsibility active set).

Still uses your existing loss utilities for:
- loading heat PLYs (we keep compatibility with bbox_json format)
- loading / printing same_pairs from relations.json
- overlap utilities as helpers

Objective (differentiable, shrink-only):
  L = w_overlap * overlap_pen + w_value * value_pen + w_same * same_pen

Notes:
- We treat "value loss" as "amount of space deleted" (volume reduction), which is differentiable.
  (This matches your simplified value loss requirement and avoids non-differentiable point-in-box counting.)
- We still load heat PLYs to keep bbox_json compatibility, but heat is not used in gradients.

Output:
- Writes optimized boxes as axis-aligned OBBs with R = identity.
- Also stores aabb_min/aabb_max in rec["opt_aabb_world"].
"""

import os
import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None

from constraints_optimization.overlap_loss import (
    obb_world_aabb_asym,
    overlap_loss_0_1,           # used for reporting
    pairwise_overlap_volume,    # not used in gradients, kept for compatibility
)

from constraints_optimization.color_value_loss import (
    load_heat_ply_points_and_heat,  # kept for bbox_json compatibility
)

from constraints_optimization.same_pair_loss import (
    load_same_pairs,
    print_same_pairs,
    same_pair_size_loss_0_1,  # used for reporting
)


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

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
        return os.path.join(os.getcwd(), "sketch", "dsl_optimize", "relations.json")
    root = p[: idx + len(marker)]  # ends with ".../sketch/"
    return os.path.join(root, "dsl_optimize", "relations.json")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_TRAILING_NUM_SUFFIX_ONCE = re.compile(r"^(.*)_(\d+)$")

def _normalize_label_base_once(label: str) -> str:
    s = str(label)
    m = _TRAILING_NUM_SUFFIX_ONCE.match(s)
    return m.group(1) if m else s

def _torch_box_volumes(mins: torch.Tensor, maxs: torch.Tensor) -> torch.Tensor:
    ext = torch.clamp(maxs - mins, min=0.0)
    return ext[:, 0] * ext[:, 1] * ext[:, 2]

def _torch_pairwise_overlap_volumes(mins: torch.Tensor, maxs: torch.Tensor) -> torch.Tensor:
    """
    mins,maxs: (N,3)
    returns overlap volume matrix (N,N) with diagonal 0.
    """
    N = mins.shape[0]
    min_i = mins.unsqueeze(1)  # (N,1,3)
    min_j = mins.unsqueeze(0)  # (1,N,3)
    max_i = maxs.unsqueeze(1)
    max_j = maxs.unsqueeze(0)
    omax = torch.minimum(max_i, max_j)
    omin = torch.maximum(min_i, min_j)
    o = F.relu(omax - omin)
    vol = o[..., 0] * o[..., 1] * o[..., 2]
    vol = vol * (1.0 - torch.eye(N, device=mins.device, dtype=mins.dtype))
    return vol

def _build_samepair_index_map(labels: List[str]) -> Dict[str, List[int]]:
    """
    base label -> list of indices (raw labels can be base_0, base_1, ...)
    """
    base_map: Dict[str, List[int]] = {}
    for i, lab in enumerate(labels):
        base = _normalize_label_base_once(lab)
        base_map.setdefault(base, []).append(i)
    return base_map

def _same_pair_penalty_torch(
    mins: torch.Tensor,
    maxs: torch.Tensor,
    same_pairs: List[Dict[str, Any]],
    base_map: Dict[str, List[int]],
    eps: float = 1e-12,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Differentiable same-pair extent mismatch penalty.

    For each (a_base, b_base), consider all raw label indices under those bases.
    Compute mismatch for each combination; use max() over combinations (subgradient ok),
    and multiply by confidence.
    """
    if not same_pairs:
        return torch.zeros((), device=mins.device, dtype=mins.dtype)

    ext = torch.clamp(maxs - mins, min=0.0)  # (N,3)
    total = torch.zeros((), device=mins.device, dtype=mins.dtype)

    for rec in same_pairs:
        a_base = str(rec.get("a", ""))
        b_base = str(rec.get("b", ""))
        w = float(rec.get("confidence", 1.0))
        if w <= 0.0:
            continue

        ia_list = base_map.get(a_base, [])
        ib_list = base_map.get(b_base, [])
        if not ia_list or not ib_list:
            continue

        ea = ext[torch.tensor(ia_list, device=mins.device)]  # (A,3)
        eb = ext[torch.tensor(ib_list, device=mins.device)]  # (B,3)

        # pairwise |ea-eb| / max(ea,eb,eps)
        ea2 = ea.unsqueeze(1)  # (A,1,3)
        eb2 = eb.unsqueeze(0)  # (1,B,3)
        denom = torch.maximum(torch.maximum(ea2, eb2), torch.tensor(float(eps), device=mins.device, dtype=mins.dtype))
        dxyz = torch.abs(ea2 - eb2) / denom

        if reduce == "max":
            d = torch.max(dxyz, dim=-1).values  # (A,B)
        else:
            d = torch.mean(dxyz, dim=-1)        # (A,B)

        worst = torch.max(d)  # scalar
        total = total + (torch.tensor(w, device=mins.device, dtype=mins.dtype) * worst)

    return total


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def apply_no_overlapping_shrink_only(
    *,
    bbox_json: str,
    out_optimized_bbox_json: str,
    out_report_json: str,
    max_iter: int = 600,
    step_frac: float = 0.08,       # kept for signature compat (ignored by Adam; see lr below)
    step_decay: float = 0.5,       # kept for signature compat (ignored)
    min_extent_frac: float = 0.15, # shrink floor relative to initial WORLD AABB extents
    w_overlap: float = 1.0,
    w_value: float = 1.0,
    w_same: float = 1.0,
    heat_gamma: float = 2.0,       # kept for signature compat (heat not used in gradients)
    print_every: int = 10,
    verbose: bool = True,
    w_shrink: float = 0.0,         # backward-compat: ignored
    device: str = "cuda",
    lr: float = 1e-2,
    overlap_tol_ratio: float = 0.02,
    overlap_scale_ratio: float = 0.01,
    **_ignored: Any,
) -> Dict[str, Any]:
    """
    Gradient-based shrink-only optimizer on WORLD AABBs.

    Parameterization per box (same as your torch code idea):
      mid0, half0 from initial WORLD AABB
      half = half0 * sigmoid(s), clamped by half_floor (derived from min_extent_frac)
      mid  = mid0 + tanh(t) * (half0 - half)   (stays inside original AABB)

    Differentiable losses:
      overlap_pen: tolerant overlap on WORLD AABB
      value_pen  : "amount of space deleted" = mean((vol0 - vol)/vol0)
      same_pen   : same-pair extent mismatch on WORLD AABB extents

    For reporting, we also compute:
      overlap_loss_0_1 (your existing normalized metric)
      same_pair_size_loss_0_1 (your existing metric)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required. Please install torch.")

    payload = load_json(bbox_json)
    labels = payload.get("labels", [])
    if not labels:
        rep = {"ok": True, "note": "No labels in bbox_json", "iters": 0}
        _save_json(out_report_json, rep)
        _save_json(out_optimized_bbox_json, payload)
        return rep

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ---- same_pairs ----
    relations_json = _infer_relations_json_from_bbox_json(bbox_json)
    same_pairs = load_same_pairs(relations_json)
    if verbose:
        print(f"[SAME_PAIR] relations_json: {relations_json}")
        print_same_pairs(same_pairs)

    # ---- build initial WORLD AABBs (one per label) ----
    # Keep heat loading for compatibility, but gradients don't use it.
    items: List[Dict[str, Any]] = []
    world_mins0: List[np.ndarray] = []
    world_maxs0: List[np.ndarray] = []

    for rec in labels:
        obb = rec.get("obb", {})
        center0 = np.asarray(obb["center"], dtype=np.float64).reshape(3)
        Rm = np.asarray(obb["R"], dtype=np.float64).reshape(3, 3)
        extent0 = np.asarray(obb["extent"], dtype=np.float64).reshape(3)

        # initial local bounds from extent (symmetric)
        half0_local = 0.5 * extent0
        bmin0 = -half0_local
        bmax0 = +half0_local

        # initial WORLD AABB from OBB+local bounds
        mn_w, mx_w = obb_world_aabb_asym(center0, Rm, bmin0, bmax0)

        # load heat ply just to validate file exists and keep format stable
        heat_ply = rec.get("heat_ply", None)
        if heat_ply is None:
            raise ValueError("Missing 'heat_ply' in bbox labels.")
        _pts_w, _heat = load_heat_ply_points_and_heat(heat_ply)

        lab = rec.get("label", rec.get("sanitized", "unknown"))
        items.append({
            "label": str(lab),
            "rec": rec,
            "center0": center0,
            "R0": Rm,
            "extent0": extent0,
            "heat_ply": heat_ply,
        })
        world_mins0.append(np.asarray(mn_w, dtype=np.float64).reshape(3))
        world_maxs0.append(np.asarray(mx_w, dtype=np.float64).reshape(3))

    mins0_t = torch.tensor(np.stack(world_mins0, axis=0), device=device, dtype=torch.float32)
    maxs0_t = torch.tensor(np.stack(world_maxs0, axis=0), device=device, dtype=torch.float32)

    # Original mid/half
    mid0 = 0.5 * (mins0_t + maxs0_t)
    half0 = 0.5 * torch.clamp(maxs0_t - mins0_t, min=1e-9)

    # Floors based on min_extent_frac
    # extent_floor_world = (maxs0-mins0) * min_extent_frac
    extent0_world = torch.clamp(maxs0_t - mins0_t, min=1e-9)
    half_floor = 0.5 * torch.clamp(extent0_world * float(min_extent_frac), min=1e-9)

    # Learnable parameters
    N = mins0_t.shape[0]
    s = torch.nn.Parameter(torch.zeros((N, 3), device=device, dtype=torch.float32))  # shrink
    t = torch.nn.Parameter(torch.zeros((N, 3), device=device, dtype=torch.float32))  # shift

    opt = torch.optim.Adam([s, t], lr=float(lr))

    # Original volumes (for "deleted space" value loss)
    vol0 = _torch_box_volumes(mins0_t, maxs0_t).detach()
    vol0 = torch.clamp(vol0, min=1e-12)

    # same-pair base mapping
    label_list = [it["label"] for it in items]
    base_map = _build_samepair_index_map(label_list)

    history: List[Dict[str, Any]] = []

    def _compute_current_mins_maxs() -> Tuple[torch.Tensor, torch.Tensor]:
        # half in (half_floor, half0] approximately
        half = half0 * torch.sigmoid(s)
        half = torch.maximum(half, half_floor)

        # shift within original bounds using slack
        slack = torch.clamp(half0 - half, min=0.0)
        mid = mid0 + torch.tanh(t) * slack

        mins = mid - half
        maxs = mid + half
        mins, maxs = torch.minimum(mins, maxs), torch.maximum(mins, maxs)
        return mins, maxs

    for itn in range(int(max_iter)):
        opt.zero_grad(set_to_none=True)

        mins, maxs = _compute_current_mins_maxs()

        # ---- differentiable overlap penalty (tolerant) ----
        vol = _torch_box_volumes(mins, maxs)
        vol = torch.clamp(vol, min=1e-12)

        V = _torch_pairwise_overlap_volumes(mins, maxs)
        Vi = vol.unsqueeze(1)
        Vj = vol.unsqueeze(0)
        Vtol = float(overlap_tol_ratio) * torch.minimum(Vi, Vj)
        excess = F.relu(V - Vtol)

        s_scale = float(overlap_scale_ratio) * torch.mean(vol).detach()
        s_scale = torch.clamp(s_scale, min=1e-12)

        overlap_pen = torch.triu(torch.log1p((excess / s_scale) ** 2), diagonal=1).sum()

        # ---- differentiable value penalty: "space deleted" ----
        # normalized per-box: (vol0 - vol)/vol0, clamped >=0
        deleted_frac = torch.clamp((vol0 - vol) / vol0, min=0.0, max=1.0)
        value_pen = torch.mean(deleted_frac)

        # ---- differentiable same-pair penalty on WORLD extents ----
        same_pen = _same_pair_penalty_torch(
            mins=mins,
            maxs=maxs,
            same_pairs=same_pairs,
            base_map=base_map,
            reduce="mean",
        )

        loss = (
            float(w_overlap) * overlap_pen
            + float(w_value) * value_pen
            + float(w_same) * same_pen
        )

        loss.backward()
        opt.step()

        # ---- reporting metrics (non-differentiable / your existing scalars) ----
        mins_np = mins.detach().cpu().numpy()
        maxs_np = maxs.detach().cpu().numpy()
        # overlap normalized metric from your overlap_loss_0_1
        ov_L, inter_sum, ov_denom, overlap_pairs, _, _ = overlap_loss_0_1(mins_np, maxs_np)

        # same normalized metric from your same_pair_size_loss_0_1
        label_to_extent = {label_list[i]: (maxs_np[i] - mins_np[i]) for i in range(len(label_list))}
        same_L = same_pair_size_loss_0_1(
            same_pairs=same_pairs,
            label_to_extent=label_to_extent,
            debug=False,
        )

        rec = {
            "iter": int(itn),
            "loss": float(loss.detach().cpu().item()),
            "overlap_pen": float(overlap_pen.detach().cpu().item()),
            "value_pen": float(value_pen.detach().cpu().item()),
            "same_pen": float(same_pen.detach().cpu().item()),
            "overlap_L": float(ov_L),
            "same_L": float(same_L),
            "overlap_pairs": int(overlap_pairs),
            "inter_sum": float(inter_sum),
            "mean_deleted": float(deleted_frac.detach().mean().cpu().item()),
            "max_deleted": float(deleted_frac.detach().max().cpu().item()),
            "min_extent": float((maxs - mins).detach().min().cpu().item()),
        }
        history.append(rec)

        if verbose and (print_every > 0) and (itn % int(print_every) == 0 or itn == int(max_iter) - 1):
            print(
                f"[NO_OVERLAP][iter={itn:04d}] "
                f"loss={rec['loss']:.6g}  "
                f"ov_pen={rec['overlap_pen']:.6g} (L={rec['overlap_L']:.4f})  "
                f"val_pen={rec['value_pen']:.6g} (mean_deleted={rec['mean_deleted']:.3f})  "
                f"same_pen={rec['same_pen']:.6g} (L={rec['same_L']:.4f})  "
                f"(pairs={rec['overlap_pairs']}, inter_sum={rec['inter_sum']:.6g}, min_ext={rec['min_extent']:.4g})"
            )

    # final mins/maxs
    with torch.no_grad():
        mins_f, maxs_f = _compute_current_mins_maxs()

    mins_f_np = mins_f.detach().cpu().numpy()
    maxs_f_np = maxs_f.detach().cpu().numpy()

    # final report metrics
    ov_L, inter_sum, ov_denom, overlap_pairs, _, _ = overlap_loss_0_1(mins_f_np, maxs_f_np)
    label_to_extent = {label_list[i]: (maxs_f_np[i] - mins_f_np[i]) for i in range(len(label_list))}
    same_L = same_pair_size_loss_0_1(
        same_pairs=same_pairs,
        label_to_extent=label_to_extent,
        debug=False,
    )

    # write optimized bbox json (axis-aligned boxes)
    out_payload = json.loads(json.dumps(payload))
    out_labels = out_payload.get("labels", [])

    I3 = np.eye(3, dtype=np.float64)

    def _write_back_axis_aligned(rec: Dict[str, Any], i: int) -> None:
        mn = mins_f_np[i].astype(np.float64)
        mx = maxs_f_np[i].astype(np.float64)
        mn2 = np.minimum(mn, mx)
        mx2 = np.maximum(mn, mx)

        c = 0.5 * (mn2 + mx2)
        e = np.maximum(mx2 - mn2, 1e-9)

        rec.setdefault("obb", {})
        rec["obb"]["center"] = c.tolist()
        rec["obb"]["extent"] = e.tolist()
        rec["obb"]["R"] = I3.tolist()  # axis-aligned

        rec["opt_aabb_world"] = {"min": mn2.tolist(), "max": mx2.tolist()}
        rec.setdefault("opt", {})
        rec["opt"].update({
            "method": "gradient_adam_world_aabb_mid_half",
            "lr": float(lr),
            "max_iter": int(max_iter),
            "weights": {"w_overlap": float(w_overlap), "w_value": float(w_value), "w_same": float(w_same)},
            "overlap_tol_ratio": float(overlap_tol_ratio),
            "overlap_scale_ratio": float(overlap_scale_ratio),
            "min_extent_frac": float(min_extent_frac),
            "note": "Optimized WORLD AABB directly; output boxes are axis-aligned (R=I).",
        })

    # assume bbox_json labels order matches items order (same as your earlier optimizer)
    if len(out_labels) == len(items):
        for i, rec in enumerate(out_labels):
            _write_back_axis_aligned(rec, i)
    else:
        # fallback: match by label string
        lab2idx = {label_list[i]: i for i in range(len(label_list))}
        for rec in out_labels:
            lab = str(rec.get("label", rec.get("sanitized", "unknown")))
            if lab in lab2idx:
                _write_back_axis_aligned(rec, lab2idx[lab])

    _save_json(out_optimized_bbox_json, out_payload)

    report = {
        "ok": True,
        "in_bbox_json": os.path.abspath(bbox_json),
        "out_bbox_json": os.path.abspath(out_optimized_bbox_json),
        "labels": int(len(items)),
        "relations_json": os.path.abspath(relations_json),
        "same_pairs": same_pairs,
        "final": {
            "overlap_L": float(ov_L),
            "same_L": float(same_L),
            "overlap_pairs": int(overlap_pairs),
            "inter_sum": float(inter_sum),
        },
        "settings": {
            "max_iter": int(max_iter),
            "lr": float(lr),
            "min_extent_frac": float(min_extent_frac),
            "overlap_tol_ratio": float(overlap_tol_ratio),
            "overlap_scale_ratio": float(overlap_scale_ratio),
            "weights": {"w_overlap": float(w_overlap), "w_value": float(w_value), "w_same": float(w_same)},
        },
        "history_tail": history[-min(200, len(history)):],
        "note": "Gradient-based Adam optimizer on WORLD AABBs. Value term is volume deleted (differentiable). Heat PLYs are loaded for compatibility but not used in gradients.",
    }
    _save_json(out_report_json, report)
    return report


def optimize_bounding_boxes(**kwargs):
    # backward-compatible alias
    return apply_no_overlapping_shrink_only(**kwargs)
