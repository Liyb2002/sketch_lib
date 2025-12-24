#!/usr/bin/env python3
import os
import json
import re
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None

try:
    import open3d as o3d
except Exception:
    o3d = None


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

def _load_primitives(primitives_json_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    raw = _load_json(primitives_json_path)
    if isinstance(raw, dict) and "primitives" in raw:
        return raw, raw["primitives"]
    if isinstance(raw, list):
        return {"primitives": raw}, raw
    raise ValueError(f"Unexpected primitives JSON format: {primitives_json_path}")


# -----------------------------------------------------------------------------
# Overlap helpers
# -----------------------------------------------------------------------------
def _pairwise_overlap_volumes(mins: torch.Tensor, maxs: torch.Tensor) -> torch.Tensor:
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
# Density / heatmap helpers (PLY-based, matches your heat_map.py outputs)
# -----------------------------------------------------------------------------
def _sanitize_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_\-]+", "", name)
    return name or "unknown"

def _heat_ply_path(heat_map_dir: str, label: str) -> str:
    s = _sanitize_name(label)
    return os.path.join(heat_map_dir, "heatmaps", s, f"heat_map_{s}.ply")

def _load_heat_ply_points_and_colors(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(path):
        return None
    if o3d is None:
        raise RuntimeError("open3d is required to read heatmap PLYs. Please `pip install open3d`.")
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    col = np.asarray(pcd.colors, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    if col.ndim != 2 or col.shape[1] != 3:
        # if somehow no colors, treat as empty heat
        col = np.zeros((pts.shape[0], 3), dtype=np.float32)
    return pts, col

def _heat_from_rgb(colors_0_1: np.ndarray) -> np.ndarray:
    c = np.clip(colors_0_1.astype(np.float32), 0.0, 1.0)
    r = c[:, 0]
    g = c[:, 1]
    h = np.where(r > 1e-6, 0.5 + 0.5 * r, 0.5 * g)
    return np.clip(h, 0.0, 1.0)

def _aabb_mask(points: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    mn = mn.astype(np.float32)
    mx = mx.astype(np.float32)
    return (
        (points[:, 0] >= mn[0]) & (points[:, 0] <= mx[0]) &
        (points[:, 1] >= mn[1]) & (points[:, 1] <= mx[1]) &
        (points[:, 2] >= mn[2]) & (points[:, 2] <= mx[2])
    )

def _aabb_mean_density_from_heatply_ptscol(pts: np.ndarray, col: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> float:
    mask = _aabb_mask(pts, mn, mx)
    if not np.any(mask):
        return 0.0
    heat = _heat_from_rgb(col[mask])
    if heat.size == 0:
        return 0.0
    v = float(np.mean(heat))
    if np.isnan(v) or np.isinf(v):
        return 0.0
    return float(np.clip(v, 0.0, 1.0))


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def _load_pcd_points(ply_path: str, max_points: int = 250000) -> np.ndarray:
    if o3d is None:
        raise RuntimeError("open3d is required for visualization. Please `pip install open3d`.")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Missing ply for vis: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts
    if max_points is not None and pts.shape[0] > int(max_points):
        idx = np.random.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[idx]
    return pts

def _make_gray_pcd(points: np.ndarray, gray: float = 0.65) -> "o3d.geometry.PointCloud":
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    col = np.full((points.shape[0], 3), float(gray), dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(col)
    return pcd

def _aabb_lineset(mn: np.ndarray, mx: np.ndarray, color_rgb=(1.0, 0.0, 0.0)) -> "o3d.geometry.LineSet":
    mn = np.asarray(mn, dtype=np.float64)
    mx = np.asarray(mx, dtype=np.float64)
    corners = np.array([
        [mn[0], mn[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]],
        [mn[0], mx[1], mx[2]],
    ], dtype=np.float64)
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)

    col = np.tile(np.array(color_rgb, dtype=np.float64)[None, :], (edges.shape[0], 1))
    ls.colors = o3d.utility.Vector3dVector(col)
    return ls

def _vis_boxes_on_shape(
    ply_path: str,
    mins_before: np.ndarray,
    maxs_before: np.ndarray,
    mins_after: np.ndarray,
    maxs_after: np.ndarray,
    *,
    title: str,
    max_boxes: int = 120,
    pcd_max_points: int = 250000,
) -> None:
    if o3d is None:
        raise RuntimeError("open3d is required for visualization. Please `pip install open3d`.")

    pts = _load_pcd_points(ply_path, max_points=pcd_max_points)
    geoms = [_make_gray_pcd(pts, gray=0.65)]

    K = mins_before.shape[0]
    if max_boxes is not None:
        K = min(K, int(max_boxes))

    for i in range(K):
        geoms.append(_aabb_lineset(mins_before[i], maxs_before[i], color_rgb=(0.1, 0.4, 1.0)))  # before
        geoms.append(_aabb_lineset(mins_after[i],  maxs_after[i],  color_rgb=(1.0, 0.2, 0.2)))  # after

    o3d.visualization.draw_geometries(geoms, window_name=title)

def _vis_boxes_per_label(
    ply_path: str,
    labels: List[str],
    label_to_indices: Dict[str, List[int]],
    mins_before: np.ndarray,
    maxs_before: np.ndarray,
    mins_after: np.ndarray,
    maxs_after: np.ndarray,
    densities: np.ndarray,
    *,
    max_labels_to_show: int,
    max_boxes_to_show: int,
    pcd_max_points: int,
) -> None:
    """
    One Open3D window per label.
    Shape is forced to uniform gray.
    Boxes: blue=before, red=after.
    Title includes mean density for that label’s boxes.
    """
    if o3d is None:
        raise RuntimeError("open3d is required for visualization. Please `pip install open3d`.")

    shown = 0
    for lab in labels:
        idxs = label_to_indices.get(lab, [])
        if not idxs:
            continue

        shown += 1
        if shown > int(max_labels_to_show):
            break

        pts = _load_pcd_points(ply_path, max_points=pcd_max_points)
        geoms = [_make_gray_pcd(pts, gray=0.65)]

        # optional: sort boxes by density (show dense ones first)
        idxs_sorted = sorted(idxs, key=lambda i: float(densities[i]) if densities is not None else 0.0, reverse=True)
        if max_boxes_to_show is not None:
            idxs_sorted = idxs_sorted[: int(max_boxes_to_show)]

        dens_vals = [float(densities[i]) for i in idxs_sorted] if densities is not None else []
        dens_mean = float(np.mean(dens_vals)) if dens_vals else 0.0

        for i in idxs_sorted:
            geoms.append(_aabb_lineset(mins_before[i], maxs_before[i], color_rgb=(0.1, 0.4, 1.0)))
            geoms.append(_aabb_lineset(mins_after[i],  maxs_after[i],  color_rgb=(1.0, 0.2, 0.2)))

        title = f"No-Overlap | label={lab} | boxes={len(idxs)} | mean_density={dens_mean:.3f} | blue=before red=after"
        o3d.visualization.draw_geometries(geoms, window_name=title)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def apply_no_overlapping_shrink_only(
    primitives_json_path: str,
    out_dir: str,
    *,
    heat_map_dir: Optional[str] = None,

    # visualization
    ply_path: Optional[str] = None,
    show_windows: bool = False,
    vis_per_label: bool = True,
    max_labels_to_show: int = 50,
    max_boxes_to_show: int = 120,
    pcd_max_points: int = 250000,

    # density weighting
    density_gamma: float = 2.0,
    density_eps: float = 0.05,
    density_default: float = 0.0,

    # optimizer
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
        e = np.abs(np.array(params.get("extent", [0, 0, 0]), dtype=np.float32))

        mn = c - 0.5 * e
        mx = c + 0.5 * e
        mn2 = np.minimum(mn, mx)
        mx2 = np.maximum(mn, mx)

        mins0.append(mn2)
        maxs0.append(mx2)
        meta.append({
            "prim_index": i,
            "cluster_id": int(p.get("cluster_id", -1)),
            "label": str(p.get("label", "unknown")),
        })

    mins0_arr = np.stack(mins0, axis=0)
    maxs0_arr = np.stack(maxs0, axis=0)

    mins0_t = torch.tensor(mins0_arr, device=device, dtype=torch.float32)
    maxs0_t = torch.tensor(maxs0_arr, device=device, dtype=torch.float32)

    mid0 = 0.5 * (mins0_t + maxs0_t)
    half0 = 0.5 * (maxs0_t - mins0_t)

    floor_half = 0.5 * float(extent_floor)
    half0 = torch.clamp(half0, min=floor_half + 1e-6)
    N = mins0_t.shape[0]

    s = torch.nn.Parameter(torch.zeros((N, 3), device=device, dtype=torch.float32))
    t = torch.nn.Parameter(torch.zeros((N, 3), device=device, dtype=torch.float32))
    opt = torch.optim.Adam([s, t], lr=lr)

    vol0 = _box_volumes(mid0 - half0, mid0 + half0).detach()
    vol0 = torch.clamp(vol0, min=1e-12)

    # -------------------------------------------------------------------------
    # Density from saved heatmap PLYs (label-specific)
    # -------------------------------------------------------------------------
    densities_np = np.full((N,), float(density_default), dtype=np.float32)

    if heat_map_dir:
        summary_path = os.path.join(heat_map_dir, "heatmaps_summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(
                f"[NO_OVERLAP][FATAL] Missing heatmaps_summary.json in heat_map_dir:\n  {summary_path}"
            )

        cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
        for k, info in enumerate(meta):
            lab = info["label"]
            if lab not in cache:
                hp = _heat_ply_path(heat_map_dir, lab)
                cache[lab] = _load_heat_ply_points_and_colors(hp)
            data = cache[lab]
            if data is None:
                densities_np[k] = float(density_default)
                continue
            pts_h, col_h = data
            densities_np[k] = _aabb_mean_density_from_heatply_ptscol(pts_h, col_h, mins0_arr[k], maxs0_arr[k])

    dens_t = torch.tensor(densities_np, device=device, dtype=torch.float32)
    shrink_resist_w = (float(density_eps) + dens_t) ** float(density_gamma)
    shrink_resist_w = torch.clamp(shrink_resist_w, min=1e-8)

    loss_hist = []

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        half = torch.clamp(half0 * torch.sigmoid(s), min=floor_half)
        slack = torch.clamp(half0 - half, min=0.0)
        mid = mid0 + torch.tanh(t) * slack

        mins = mid - half
        maxs = mid + half
        mins, maxs = torch.minimum(mins, maxs), torch.maximum(mins, maxs)

        extent = maxs - mins
        floor_pen = torch.mean(F.relu(float(extent_floor) - extent) ** 2)

        vol = torch.clamp(_box_volumes(mins, maxs), min=1e-12)
        r = vol / vol0

        cut_pen = torch.mean(shrink_resist_w * (F.relu(float(r_min) - r) ** 2))
        size_pen = torch.mean(shrink_resist_w * (1.0 - r))

        V = _pairwise_overlap_volumes(mins, maxs)
        Vi = vol.unsqueeze(1)
        Vj = vol.unsqueeze(0)
        Vtol = float(overlap_tol_ratio) * torch.minimum(Vi, Vj)
        excess = F.relu(V - Vtol)

        s_scale = torch.clamp(float(overlap_scale_ratio) * torch.mean(vol).detach(), min=1e-12)
        overlap_pen = torch.triu(torch.log1p((excess / s_scale) ** 2), diagonal=1).sum()

        loss = w_overlap * overlap_pen + w_cut * cut_pen + w_size * size_pen + w_floor * floor_pen
        loss.backward()
        opt.step()

        if verbose_every > 0 and (step % verbose_every == 0 or step == steps - 1):
            print(
                f"[NO_OVERLAP] step={step:04d} "
                f"loss={float(loss.detach().cpu().item()):.6e} "
                f"overlap={float(overlap_pen.detach().cpu().item()):.6e} "
                f"mean_r={float(r.detach().mean().cpu().item()):.3f} "
                f"min_r={float(r.detach().min().cpu().item()):.3f} "
                f"mean_density={float(dens_t.mean().detach().cpu().item()):.3f}"
            )

        if step >= steps - 120:
            loss_hist.append({
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "overlap_pen": float(overlap_pen.detach().cpu().item()),
                "cut_pen": float(cut_pen.detach().cpu().item()),
                "size_pen": float(size_pen.detach().cpu().item()),
                "floor_pen": float(floor_pen.detach().cpu().item()),
                "mean_r": float(r.detach().mean().cpu().item()),
                "min_r": float(r.detach().min().cpu().item()),
                "mean_density": float(dens_t.mean().detach().cpu().item()),
            })

    with torch.no_grad():
        half = torch.clamp(half0 * torch.sigmoid(s), min=floor_half)
        slack = torch.clamp(half0 - half, min=0.0)
        mid = mid0 + torch.tanh(t) * slack
        mins = torch.minimum(mid - half, mid + half)
        maxs = torch.maximum(mid - half, mid + half)

    mins_opt = mins.detach().cpu().numpy()
    maxs_opt = maxs.detach().cpu().numpy()

    # -------------------------------------------------------------------------
    # VIS: one window per label (requested)
    # -------------------------------------------------------------------------
    if show_windows and ply_path is not None:
        try:
            labels = [m["label"] for m in meta]
            uniq_labels = sorted(set(labels))

            label_to_indices: Dict[str, List[int]] = {}
            for i, lab in enumerate(labels):
                label_to_indices.setdefault(lab, []).append(i)

            if vis_per_label:
                _vis_boxes_per_label(
                    ply_path=ply_path,
                    labels=uniq_labels,
                    label_to_indices=label_to_indices,
                    mins_before=mins0_arr,
                    maxs_before=maxs0_arr,
                    mins_after=mins_opt,
                    maxs_after=maxs_opt,
                    densities=densities_np,
                    max_labels_to_show=max_labels_to_show,
                    max_boxes_to_show=max_boxes_to_show,
                    pcd_max_points=pcd_max_points,
                )
            else:
                _vis_boxes_on_shape(
                    ply_path=ply_path,
                    mins_before=mins0_arr,
                    maxs_before=maxs0_arr,
                    mins_after=mins_opt,
                    maxs_after=maxs_opt,
                    title="No-Overlap | blue=before | red=after",
                    max_boxes=max_boxes_to_show,
                    pcd_max_points=pcd_max_points,
                )
        except Exception as e:
            print(f"[NO_OVERLAP][VIS] Warning: visualization failed: {e}")

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------
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
        mn2 = np.minimum(mn, mx)
        mx2 = np.maximum(mn, mx)

        c = 0.5 * (mn2 + mx2)
        e = np.maximum((mx2 - mn2), extent_floor)

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
            "density": float(densities_np[k]) if k < len(densities_np) else None,
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
        "method": "no_overlapping_shrink_only_density_aware_mid_half",
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
            "heat_map_dir": os.path.abspath(heat_map_dir) if heat_map_dir else None,
            "density_gamma": float(density_gamma),
            "density_eps": float(density_eps),
            "density_default": float(density_default),
            "show_windows": bool(show_windows),
            "vis_per_label": bool(vis_per_label),
            "max_labels_to_show": int(max_labels_to_show),
            "max_boxes_to_show": int(max_boxes_to_show),
            "pcd_max_points": int(pcd_max_points),
        },
        "weights": {
            "w_overlap": float(w_overlap),
            "w_cut": float(w_cut),
            "w_size": float(w_size),
            "w_floor": float(w_floor),
        },
        "note": "Density-aware cut/size penalties weighted by mean heat inside original AABB using saved per-label heatmap PLYs. Vis uses uniform-gray shape."
    })

    _save_json(out_primitives_path, out_raw)
    _save_json(out_report_path, {
        "inputs": {"primitives_json": os.path.abspath(primitives_json_path)},
        "settings": out_raw["optimization"],
        "density_stats": {
            "count": int(N),
            "mean_density": float(np.mean(densities_np)) if len(densities_np) else None,
            "min_density": float(np.min(densities_np)) if len(densities_np) else None,
            "max_density": float(np.max(densities_np)) if len(densities_np) else None,
        },
        "changed_summary": changed,
        "loss_history_tail": loss_hist,
    })

    return {"optimized_primitives_json": out_primitives_path, "report_json": out_report_path}
