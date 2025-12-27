#!/usr/bin/env python3
"""
constraints_optimization/optimizer.py

Adam shrink-only optimizer over WORLD AABB boxes.

Key change (per your request):
- Value loss is computed as EXACT "points in the cut area" (hard inside/outside):
    removed_i = sum(value_of_point for points that were inside initially, but outside now)

But to keep gradients for Adam:
- We use STE (straight-through estimator):
    forward uses hard inside test,
    backward uses soft inside test.

So:
- Iter 0 removed == 0 (if initial box contains all colored points)
- During optimization removed is exact sum of removed point values.
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
    overlap_loss_0_1,
)

from constraints_optimization.color_value_loss import (
    load_heat_ply_points_and_heat,
)

from constraints_optimization.same_pair_loss import (
    load_same_pairs,
    print_same_pairs,
    same_pair_size_loss_0_1,
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
    p = os.path.abspath(bbox_json)
    marker = os.sep + "sketch" + os.sep
    idx = p.rfind(marker)
    if idx < 0:
        return os.path.join(os.getcwd(), "sketch", "dsl_optimize", "relations.json")
    root = p[: idx + len(marker)]
    return os.path.join(root, "dsl_optimize", "relations.json")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_TRAILING_NUM_SUFFIX_ONCE = re.compile(r"^(.*)_(\d+)$")

def _normalize_label_base_once(label: str) -> str:
    s = str(label)
    m = _TRAILING_NUM_SUFFIX_ONCE.match(s)
    return m.group(1) if m else s

def _build_samepair_index_map(labels: List[str]) -> Dict[str, List[int]]:
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

        ea2 = ea.unsqueeze(1)  # (A,1,3)
        eb2 = eb.unsqueeze(0)  # (1,B,3)
        denom = torch.maximum(torch.maximum(ea2, eb2), torch.tensor(float(eps), device=mins.device, dtype=mins.dtype))
        dxyz = torch.abs(ea2 - eb2) / denom

        if reduce == "max":
            d = torch.max(dxyz, dim=-1).values  # (A,B)
        else:
            d = torch.mean(dxyz, dim=-1)        # (A,B)

        worst = torch.max(d)
        total = total + (torch.tensor(w, device=mins.device, dtype=mins.dtype) * worst)

    return total


# -----------------------------------------------------------------------------
# Overlap penalty (differentiable, tolerant, averaged over pairs)
# -----------------------------------------------------------------------------

def _torch_pairwise_overlap_volumes(mins: torch.Tensor, maxs: torch.Tensor) -> torch.Tensor:
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

def _torch_box_volumes(mins: torch.Tensor, maxs: torch.Tensor) -> torch.Tensor:
    ext = torch.clamp(maxs - mins, min=0.0)
    return ext[:, 0] * ext[:, 1] * ext[:, 2]


# -----------------------------------------------------------------------------
# Hard + Soft inside tests and STE
# -----------------------------------------------------------------------------

def _hard_inside(points_w: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
    """
    Hard membership in {0,1}: (M,)
    """
    ge = points_w >= mn[None, :]
    le = points_w <= mx[None, :]
    m = (ge & le).all(dim=1)
    return m.to(dtype=points_w.dtype)

def _soft_inside(points_w: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Soft membership in [0,1]: (M,)
    """
    left = torch.sigmoid((points_w - mn[None, :]) / tau)
    right = torch.sigmoid((mx[None, :] - points_w) / tau)
    w = left * right
    return w[:, 0] * w[:, 1] * w[:, 2]

def _ste_inside(points_w: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Straight-through estimator:
    - forward: hard inside
    - backward: soft inside
    """
    hard = _hard_inside(points_w, mn, mx)
    soft = _soft_inside(points_w, mn, mx, tau=tau)
    # forward value = hard, gradient = soft
    return soft + (hard - soft).detach()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def apply_no_overlapping_shrink_only(
    *,
    bbox_json: str,
    out_optimized_bbox_json: str,
    out_report_json: str,
    max_iter: int = 600,
    min_extent_frac: float = 0.15,
    w_overlap: float = 1.0,
    w_value: float = 1.0,
    w_same: float = 1.0,
    print_every: int = 10,
    verbose: bool = True,
    device: str = "cuda",
    lr: float = 1e-2,
    overlap_tol_ratio: float = 0.02,
    overlap_scale_ratio: float = 0.01,
    soft_tau_frac: float = 0.002,  # softness for STE backward; can go smaller if stable
    **_ignored: Any,
) -> Dict[str, Any]:
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

    # ---- build initial WORLD AABBs + load per-label heat points ----
    items: List[Dict[str, Any]] = []
    world_mins0: List[np.ndarray] = []
    world_maxs0: List[np.ndarray] = []
    pts_w_all: List[np.ndarray] = []
    heat_all: List[np.ndarray] = []

    for rec in labels:
        obb = rec.get("obb", {})
        center0 = np.asarray(obb["center"], dtype=np.float64).reshape(3)
        Rm = np.asarray(obb["R"], dtype=np.float64).reshape(3, 3)
        extent0 = np.asarray(obb["extent"], dtype=np.float64).reshape(3)

        half0_local = 0.5 * extent0
        bmin0 = -half0_local
        bmax0 = +half0_local
        mn_w, mx_w = obb_world_aabb_asym(center0, Rm, bmin0, bmax0)

        heat_ply = rec.get("heat_ply", None)
        if heat_ply is None:
            raise ValueError("Missing 'heat_ply' in bbox labels.")
        pts_w, heat = load_heat_ply_points_and_heat(heat_ply)

        lab = rec.get("label", rec.get("sanitized", "unknown"))
        items.append({"label": str(lab), "rec": rec, "heat_ply": heat_ply})
        world_mins0.append(np.asarray(mn_w, dtype=np.float64).reshape(3))
        world_maxs0.append(np.asarray(mx_w, dtype=np.float64).reshape(3))
        pts_w_all.append(np.asarray(pts_w, dtype=np.float32))
        heat_all.append(np.asarray(heat, dtype=np.float32).reshape(-1))

    mins0_t = torch.tensor(np.stack(world_mins0, axis=0), device=device, dtype=torch.float32)
    maxs0_t = torch.tensor(np.stack(world_maxs0, axis=0), device=device, dtype=torch.float32)

    # Parameterization: mid/half with shrink + shift (shrink-only inside original)
    mid0 = 0.5 * (mins0_t + maxs0_t)
    half0 = 0.5 * torch.clamp(maxs0_t - mins0_t, min=1e-9)

    extent0_world = torch.clamp(maxs0_t - mins0_t, min=1e-9)
    half_floor = 0.5 * torch.clamp(extent0_world * float(min_extent_frac), min=1e-9)

    N = mins0_t.shape[0]
    s = torch.nn.Parameter(torch.zeros((N, 3), device=device, dtype=torch.float32))  # shrink
    t = torch.nn.Parameter(torch.zeros((N, 3), device=device, dtype=torch.float32))  # shift
    opt = torch.optim.Adam([s, t], lr=float(lr))

    label_list = [it["label"] for it in items]
    base_map = _build_samepair_index_map(label_list)

    # Load tensors; keep ONLY valued points (heat>0) for speed and correctness
    pts_t: List[torch.Tensor] = []
    heat_t: List[torch.Tensor] = []
    for p, h in zip(pts_w_all, heat_all):
        if p.size == 0 or h.size == 0:
            pts_t.append(torch.zeros((0, 3), device=device, dtype=torch.float32))
            heat_t.append(torch.zeros((0,), device=device, dtype=torch.float32))
            continue
        m = h > 0.0
        p2 = p[m]
        h2 = h[m]
        pts_t.append(torch.tensor(p2, device=device, dtype=torch.float32))
        heat_t.append(torch.tensor(h2, device=device, dtype=torch.float32))

    # Baseline: hard "inside initial box" membership (should be all ones if your bbox was built over colored points)
    # value0_i = sum of values of points that are inside initial bbox
    with torch.no_grad():
        value0_list = []
        for i in range(N):
            if pts_t[i].numel() == 0:
                value0_list.append(torch.zeros((), device=device, dtype=torch.float32))
            else:
                inside0 = _hard_inside(pts_t[i], mins0_t[i], maxs0_t[i])
                value0_list.append(torch.sum(heat_t[i] * inside0))
        value0 = torch.stack(value0_list, dim=0)  # (N,)
        value0 = torch.clamp(value0, min=1e-12)
    sum_value0 = torch.clamp(torch.sum(value0), min=1e-12)

    # STE softness scale
    mean_extent = torch.mean(extent0_world).detach()
    tau = float(max(1e-8, float(soft_tau_frac))) * float(mean_extent.cpu().item())
    tau = float(max(1e-6, tau))

    def _compute_current_mins_maxs() -> Tuple[torch.Tensor, torch.Tensor]:
        half = half0 * torch.sigmoid(s)
        half = torch.maximum(half, half_floor)
        slack = torch.clamp(half0 - half, min=0.0)
        mid = mid0 + torch.tanh(t) * slack
        mn = mid - half
        mx = mid + half
        mn, mx = torch.minimum(mn, mx), torch.maximum(mn, mx)
        return mn, mx

    history: List[Dict[str, Any]] = []

    for itn in range(int(max_iter)):
        opt.zero_grad(set_to_none=True)
        mins, maxs = _compute_current_mins_maxs()

        # ---- overlap penalty (tolerant, averaged over pairs) ----
        vol = torch.clamp(_torch_box_volumes(mins, maxs), min=1e-12)
        V = _torch_pairwise_overlap_volumes(mins, maxs)
        Vi = vol.unsqueeze(1)
        Vj = vol.unsqueeze(0)
        Vtol = float(overlap_tol_ratio) * torch.minimum(Vi, Vj)
        excess = F.relu(V - Vtol)

        s_scale = float(overlap_scale_ratio) * torch.mean(vol).detach()
        s_scale = torch.clamp(s_scale, min=1e-12)

        per_pair = torch.log1p(excess / s_scale)
        mask = torch.triu(torch.ones((N, N), device=device, dtype=torch.float32), diagonal=1)
        pair_count = torch.clamp(mask.sum(), min=1.0)
        overlap_pen = (per_pair * mask).sum() / pair_count

        # ---- value penalty: EXACT cut points (forward), STE gradients (backward) ----
        removed_list = []
        for i in range(N):
            if pts_t[i].numel() == 0:
                removed_list.append(torch.zeros((), device=device, dtype=torch.float32))
                continue

            # points that were inside initially (hard)
            inside0 = _hard_inside(pts_t[i], mins0_t[i], maxs0_t[i])  # (Mi,)
            # current inside with STE: forward hard, backward soft
            inside_now = _ste_inside(pts_t[i], mins[i], maxs[i], tau=tau)  # (Mi,)

            # removed = points that were inside0 but are NOT inside now
            removed_mask = inside0 * (1.0 - inside_now)  # forward uses exact hard inside_now
            removed_val = torch.sum(heat_t[i] * removed_mask)
            removed_list.append(removed_val)

        removed_vals = torch.stack(removed_list, dim=0)  # (N,)
        sum_removed = torch.sum(removed_vals)
        value_pen = torch.clamp(sum_removed / sum_value0, 0.0, 1.0)

        # ---- same-pair penalty ----
        same_pen = _same_pair_penalty_torch(mins, maxs, same_pairs, base_map, reduce="mean")

        loss = float(w_overlap) * overlap_pen + float(w_value) * value_pen + float(w_same) * same_pen
        loss.backward()
        opt.step()

        # ---- reporting (your normalized overlap + same) ----
        mins_np = mins.detach().cpu().numpy()
        maxs_np = maxs.detach().cpu().numpy()

        ov_L, inter_sum, _, overlap_pairs, _, _ = overlap_loss_0_1(mins_np, maxs_np)

        label_to_extent = {label_list[i]: (maxs_np[i] - mins_np[i]) for i in range(len(label_list))}
        same_L = same_pair_size_loss_0_1(same_pairs=same_pairs, label_to_extent=label_to_extent, debug=False)

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
            "sum_removed": float(sum_removed.detach().cpu().item()),
            "sum_value0": float(sum_value0.detach().cpu().item()),
            "min_extent": float((maxs - mins).detach().min().cpu().item()),
            "tau": float(tau),
        }
        history.append(rec)

        if verbose and (print_every > 0) and (itn % int(print_every) == 0 or itn == int(max_iter) - 1):
            print(
                f"[NO_OVERLAP][iter={itn:04d}] "
                f"loss={rec['loss']:.6g}  "
                f"ov_pen={rec['overlap_pen']:.6g} (L={rec['overlap_L']:.4f})  "
                f"val_pen={rec['value_pen']:.6g} (sum_removed={rec['sum_removed']:.6g})  "
                f"same_pen={rec['same_pen']:.6g} (L={rec['same_L']:.4f})  "
                f"(pairs={rec['overlap_pairs']}, inter_sum={rec['inter_sum']:.6g}, min_ext={rec['min_extent']:.4g})"
            )

    # final mins/maxs
    with torch.no_grad():
        mins_f, maxs_f = _compute_current_mins_maxs()

    mins_f_np = mins_f.detach().cpu().numpy()
    maxs_f_np = maxs_f.detach().cpu().numpy()

    # final report metrics
    ov_L, inter_sum, _, overlap_pairs, _, _ = overlap_loss_0_1(mins_f_np, maxs_f_np)
    label_to_extent = {label_list[i]: (maxs_f_np[i] - mins_f_np[i]) for i in range(len(label_list))}
    same_L = same_pair_size_loss_0_1(same_pairs=same_pairs, label_to_extent=label_to_extent, debug=False)

    # write optimized bbox json (axis-aligned output)
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
        rec["obb"]["R"] = I3.tolist()
        rec["opt_aabb_world"] = {"min": mn2.tolist(), "max": mx2.tolist()}
        rec.setdefault("opt", {})
        rec["opt"].update({
            "method": "adam_world_aabb_value_exact_cutpoints_STE",
            "lr": float(lr),
            "max_iter": int(max_iter),
            "weights": {"w_overlap": float(w_overlap), "w_value": float(w_value), "w_same": float(w_same)},
            "min_extent_frac": float(min_extent_frac),
            "soft_tau_frac": float(soft_tau_frac),
            "tau_world": float(tau),
            "note": "Value loss forward is exact sum of cut point values (points inside initial box but outside now). Gradients via STE using soft inside.",
        })

    if len(out_labels) == len(items):
        for i, rec in enumerate(out_labels):
            _write_back_axis_aligned(rec, i)
    else:
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
            "soft_tau_frac": float(soft_tau_frac),
            "tau_world": float(tau),
            "weights": {"w_overlap": float(w_overlap), "w_value": float(w_value), "w_same": float(w_same)},
        },
        "history_tail": history[-min(200, len(history)):],
        "note": "Value loss is exact cut-point value sum (forward). Adam uses STE gradients (soft inside backward).",
    }
    _save_json(out_report_json, report)
    return report


def optimize_bounding_boxes(**kwargs):
    return apply_no_overlapping_shrink_only(**kwargs)
