#!/usr/bin/env python3
import os
import json
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _load_primitives(primitives_json_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    raw = _load_json(primitives_json_path)
    if isinstance(raw, dict) and "primitives" in raw:
        return raw, raw["primitives"]
    if isinstance(raw, list):
        return {"primitives": raw}, raw
    raise ValueError(f"Unexpected primitives JSON format: {primitives_json_path}")


# ---------------- overlap ----------------

def _pairwise_overlap_volumes(mins: torch.Tensor, maxs: torch.Tensor) -> torch.Tensor:
    """
    mins,maxs: (N,3)
    returns overlap volume matrix (N,N) with diagonal 0.
    """
    N = mins.shape[0]
    min_i = mins.unsqueeze(1)
    min_j = mins.unsqueeze(0)
    max_i = maxs.unsqueeze(1)
    max_j = maxs.unsqueeze(0)
    omax = torch.minimum(max_i, max_j)
    omin = torch.maximum(min_i, min_j)
    o = F.relu(omax - omin)
    vol = o[..., 0] * o[..., 1] * o[..., 2]
    vol = vol * (1.0 - torch.eye(N, device=mins.device, dtype=mins.dtype))
    return vol

def _box_volumes(mins: torch.Tensor, maxs: torch.Tensor) -> torch.Tensor:
    ext = torch.clamp(maxs - mins, min=0.0)
    return ext[:, 0] * ext[:, 1] * ext[:, 2]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def apply_no_overlapping_shrink_only(
    primitives_json_path: str,
    out_dir: str,
    *,
    steps: int = 1200,
    lr: float = 1e-2,
    overlap_tol_ratio: float = 0.02,
    overlap_scale_ratio: float = 0.01,
    r_min: float = 0.70,
    w_overlap: float = 1.0,
    w_cut: float = 50.0,
    w_size: float = 2.0,
    w_floor: float = 10.0,
    extent_floor: float = 1e-3,
    min_points: int = 10,
    device: str = "cuda",
    verbose_every: int = 50,
) -> Dict[str, Any]:
    """
    Robust shrink-only optimizer that NEVER produces negative extents.

    Parameterization:
      - original AABB: mid0, half0
      - half = half0 * sigmoid(s) + floor_half   (positive, <= half0+floor_half)
      - mid  = mid0 + tanh(t) * (half0 - half)   (keeps box inside original)

    So mins = mid-half, maxs = mid+half are always ordered, extent=2*half always positive.

    Loss:
      - tolerant nonlinear overlap
      - penalize too much cut (vol ratio below r_min)
      - gentle size encouragement
      - floor penalty (mostly redundant, but stabilizes)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required. Please install torch.")

    os.makedirs(out_dir, exist_ok=True)
    raw, prims = _load_primitives(primitives_json_path)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    eligible_idxs = [i for i, p in enumerate(prims) if int(p.get("point_count", 0)) >= min_points]
    if len(eligible_idxs) <= 1:
        out_path = os.path.join(out_dir, "optimized_primitives.json")
        rep_path = os.path.join(out_dir, "no_overlapping_report.json")
        out_raw = dict(raw) if isinstance(raw, dict) else {"primitives": prims}
        out_raw["primitives"] = prims
        _save_json(out_path, out_raw)
        _save_json(rep_path, {"note": "Not enough eligible primitives.", "eligible_count": len(eligible_idxs)})
        return {"optimized_primitives_json": out_path, "report_json": rep_path}

    mins0, maxs0, meta = [], [], []
    for i in eligible_idxs:
        p = prims[i]
        params = p.get("parameters", {})
        c = np.array(params.get("center", [0, 0, 0]), dtype=np.float32)
        e = np.array(params.get("extent", [0, 0, 0]), dtype=np.float32)

        # sanitize original extent (in case upstream has negatives)
        e = np.abs(e)

        mn = c - 0.5 * e
        mx = c + 0.5 * e

        # ensure ordered original bounds (paranoia)
        mn2 = np.minimum(mn, mx)
        mx2 = np.maximum(mn, mx)

        mins0.append(mn2)
        maxs0.append(mx2)
        meta.append({
            "prim_index": i,
            "cluster_id": int(p.get("cluster_id", -1)),
            "label": str(p.get("label", "unknown")),
        })

    mins0_t = torch.tensor(np.stack(mins0, axis=0), device=device, dtype=torch.float32)
    maxs0_t = torch.tensor(np.stack(maxs0, axis=0), device=device, dtype=torch.float32)

    mid0 = 0.5 * (mins0_t + maxs0_t)
    half0 = 0.5 * (maxs0_t - mins0_t)

    # enforce a minimum half extent floor
    floor_half = 0.5 * float(extent_floor)
    half0 = torch.clamp(half0, min=floor_half + 1e-6)

    N = mins0_t.shape[0]

    # learnable params
    s = torch.nn.Parameter(torch.zeros((N, 3), device=device, dtype=torch.float32))  # controls shrink
    t = torch.nn.Parameter(torch.zeros((N, 3), device=device, dtype=torch.float32))  # controls shift within original

    opt = torch.optim.Adam([s, t], lr=lr)

    # compute original volumes
    mins_init = mid0 - half0
    maxs_init = mid0 + half0
    vol0 = _box_volumes(mins_init, maxs_init).detach()
    vol0 = torch.clamp(vol0, min=1e-12)

    loss_hist = []

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        # half in (floor_half, half0] (approx)
        half = half0 * torch.sigmoid(s)
        half = torch.clamp(half, min=floor_half)

        # allow mid shift only as much as still staying inside original bounds:
        # available slack = half0 - half (>=0)
        slack = torch.clamp(half0 - half, min=0.0)
        mid = mid0 + torch.tanh(t) * slack

        mins = mid - half
        maxs = mid + half

        # safety ordering (should be unnecessary, but guarantees no negatives)
        mins, maxs = torch.minimum(mins, maxs), torch.maximum(mins, maxs)

        extent = maxs - mins
        floor_pen = torch.mean(F.relu(float(extent_floor) - extent) ** 2)

        vol = _box_volumes(mins, maxs)
        vol = torch.clamp(vol, min=1e-12)
        r = vol / vol0

        cut_pen = torch.mean(F.relu(float(r_min) - r) ** 2)
        size_pen = torch.mean(1.0 - r)

        V = _pairwise_overlap_volumes(mins, maxs)
        Vi = vol.unsqueeze(1)
        Vj = vol.unsqueeze(0)
        Vtol = float(overlap_tol_ratio) * torch.minimum(Vi, Vj)
        excess = F.relu(V - Vtol)

        s_scale = float(overlap_scale_ratio) * torch.mean(vol).detach()
        s_scale = torch.clamp(s_scale, min=1e-12)

        overlap_pen = torch.triu(torch.log1p((excess / s_scale) ** 2), diagonal=1).sum()

        loss = w_overlap * overlap_pen + w_cut * cut_pen + w_size * size_pen + w_floor * floor_pen
        loss.backward()
        opt.step()

        rec = {
            "step": step,
            "loss": float(loss.detach().cpu().item()),
            "overlap_pen": float(overlap_pen.detach().cpu().item()),
            "cut_pen": float(cut_pen.detach().cpu().item()),
            "size_pen": float(size_pen.detach().cpu().item()),
            "floor_pen": float(floor_pen.detach().cpu().item()),
            "mean_r": float(r.detach().mean().cpu().item()),
            "min_r": float(r.detach().min().cpu().item()),
            "min_extent": float(extent.detach().min().cpu().item()),
        }
        loss_hist.append(rec)

        if verbose_every > 0 and (step % verbose_every == 0 or step == steps - 1):
            print(
                f"[NO_OVERLAP] step={step:04d} "
                f"loss={rec['loss']:.6e} "
                f"overlap={rec['overlap_pen']:.6e} "
                f"cut={rec['cut_pen']:.6e} "
                f"mean_r={rec['mean_r']:.3f} "
                f"min_r={rec['min_r']:.3f} "
                f"min_extent={rec['min_extent']:.3e}"
            )

    with torch.no_grad():
        half = torch.clamp(half0 * torch.sigmoid(s), min=floor_half)
        slack = torch.clamp(half0 - half, min=0.0)
        mid = mid0 + torch.tanh(t) * slack
        mins = torch.minimum(mid - half, mid + half)
        maxs = torch.maximum(mid - half, mid + half)

    mins_opt = mins.detach().cpu().numpy()
    maxs_opt = maxs.detach().cpu().numpy()

    new_prims = json.loads(json.dumps(prims))
    changed = []

    for k, info in enumerate(meta):
        pi = info["prim_index"]
        p_old = prims[pi]
        p_new = new_prims[pi]
        params_old = p_old.get("parameters", {})
        params_new = p_new.setdefault("parameters", {})

        c0 = np.array(params_old.get("center", [0, 0, 0]), dtype=np.float64)
        e0 = np.abs(np.array(params_old.get("extent", [0, 0, 0]), dtype=np.float64))

        mn0 = np.minimum(c0 - 0.5 * e0, c0 + 0.5 * e0)
        mx0 = np.maximum(c0 - 0.5 * e0, c0 + 0.5 * e0)

        mn = mins_opt[k].astype(np.float64)
        mx = maxs_opt[k].astype(np.float64)

        # final safety ordering
        mn2 = np.minimum(mn, mx)
        mx2 = np.maximum(mn, mx)

        c = 0.5 * (mn2 + mx2)
        e = (mx2 - mn2)
        e = np.maximum(e, extent_floor)  # hard clamp: cannot be negative

        params_new["aabb_min_before_opt"] = mn0.tolist()
        params_new["aabb_max_before_opt"] = mx0.tolist()
        params_new["aabb_min"] = mn2.tolist()
        params_new["aabb_max"] = mx2.tolist()

        params_new["center_before_opt"] = params_old.get("center", [0, 0, 0])
        params_new["extent_before_opt"] = params_old.get("extent", [0, 0, 0])
        params_new["center"] = c.tolist()
        params_new["extent"] = e.tolist()

        v0 = float(np.prod(np.maximum(e0, 1e-12)))
        v1 = float(np.prod(np.maximum(e, 1e-12)))
        changed.append({
            "cluster_id": int(p_old.get("cluster_id", -1)),
            "label": str(p_old.get("label", "unknown")),
            "vol_ratio": (v1 / v0) if v0 > 0 else None,
            "extent_before": params_old.get("extent", [0, 0, 0]),
            "extent_after": e.tolist(),
        })

    out_primitives_path = os.path.join(out_dir, "optimized_primitives.json")
    out_report_path = os.path.join(out_dir, "no_overlapping_report.json")

    out_raw = dict(raw) if isinstance(raw, dict) else {"primitives": new_prims}
    out_raw["primitives"] = new_prims
    out_raw.setdefault("optimization", {})
    out_raw["optimization"].update({
        "method": "no_overlapping_shrink_only_size_max_mid_half",
        "never_expand": True,
        "params": {
            "steps": int(steps),
            "lr": float(lr),
            "overlap_tol_ratio": float(overlap_tol_ratio),
            "overlap_scale_ratio": float(overlap_scale_ratio),
            "r_min": float(r_min),
            "extent_floor": float(extent_floor),
            "min_points": int(min_points),
            "device": str(device),
        },
        "weights": {
            "w_overlap": float(w_overlap),
            "w_cut": float(w_cut),
            "w_size": float(w_size),
            "w_floor": float(w_floor),
        },
        "note": "Uses mid/half parameterization to guarantee non-negative extents and ordered mins/maxs."
    })

    _save_json(out_primitives_path, out_raw)
    _save_json(out_report_path, {
        "inputs": {"primitives_json": os.path.abspath(primitives_json_path)},
        "settings": out_raw["optimization"],
        "changed_summary": changed,
        "loss_history_tail": loss_hist[-min(120, len(loss_hist)):],
    })

    return {"optimized_primitives_json": out_primitives_path, "report_json": out_report_path}
