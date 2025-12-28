#!/usr/bin/env python3
"""
13_read_changes.py

AEP edit step (placeholder, drag-like):
- Randomly pick a target bbox label (exclude labels starting with "unknown_")
- Edit by "dragging" ONE WORLD-AABB face (min/max of x/y/z) by 10% to 50% of that axis length
- Save edited bbox info to: sketch/target_edit/

Visualization:
1) One window: BEFORE + AFTER in the SAME view (AABB)
   - BEFORE = blue
   - AFTER  = red (stronger)
   - faint fused shape overlay

2) Another window: show BEFORE+AFTER (AABB) with all its relations + anchor points
   - neighbor bboxes shown (from graph, unchanged)
   - anchor points/segments shown ONCE, computed from BEFORE (graph) AABBs only (NOT recomputed)
   - faint fused shape overlay

Hardcoded iteration: iter_000
"""

import os
import json
import random
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# ------------------------ IO ------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_txt(path: str, s: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(s)


# ------------------------ helpers ------------------------

def _np3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)

def _np33(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3, 3)

def _deepcopy_jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x))


# ------------------------ OBB -> WORLD AABB ------------------------

def _obb_world_aabb(center: np.ndarray, R: np.ndarray, extent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust WORLD AABB for an OBB: transform 8 corners.
    """
    c = center.reshape(3)
    R = R.reshape(3, 3)
    e = extent.reshape(3)

    half = 0.5 * e
    xs = [-half[0], half[0]]
    ys = [-half[1], half[1]]
    zs = [-half[2], half[2]]
    corners_local = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    corners_world = (R @ corners_local.T).T + c[None, :]
    return corners_world.min(axis=0), corners_world.max(axis=0)


# ------------------------ AABB anchors (BEFORE only) ------------------------

def _aabb_closest_points(
    mnA: np.ndarray, mxA: np.ndarray,
    mnB: np.ndarray, mxB: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute closest points pA (on/inside A) and pB (on/inside B), axis-wise.
    Pin is midpoint between pA and pB.
    """
    pA = np.zeros((3,), dtype=np.float64)
    pB = np.zeros((3,), dtype=np.float64)
    for k in range(3):
        if mxA[k] < mnB[k]:
            pA[k] = mxA[k]
            pB[k] = mnB[k]
        elif mxB[k] < mnA[k]:
            pA[k] = mnA[k]
            pB[k] = mxB[k]
        else:
            lo = max(mnA[k], mnB[k])
            hi = min(mxA[k], mxB[k])
            mid = 0.5 * (lo + hi)
            pA[k] = mid
            pB[k] = mid
    pin = 0.5 * (pA + pB)
    return pA, pB, pin


# ------------------------ Open3D geometry ------------------------

def _load_fused_geometry_faint(fused_ply: str) -> List["o3d.geometry.Geometry"]:
    if o3d is None:
        raise RuntimeError("Open3D not available.")
    if not os.path.isfile(fused_ply):
        raise FileNotFoundError(f"Missing fused model ply: {fused_ply}")

    pcd = o3d.io.read_point_cloud(fused_ply)
    if pcd is not None and len(pcd.points) > 0:
        pcd.paint_uniform_color([0.75, 0.75, 0.75])
        return [pcd]

    mesh = o3d.io.read_triangle_mesh(fused_ply)
    if mesh is not None and len(mesh.vertices) > 0:
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.75, 0.75, 0.75])
        return [mesh]

    raise RuntimeError(f"Loaded fused_model.ply but empty: {fused_ply}")

def _make_aabb_lineset(mn: np.ndarray, mx: np.ndarray) -> "o3d.geometry.LineSet":
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=mn.reshape(3), max_bound=mx.reshape(3))
    ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    return ls

def _make_pin_sphere(center: np.ndarray, radius: float) -> "o3d.geometry.TriangleMesh":
    m = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
    m.translate(center.reshape(3))
    m.compute_vertex_normals()
    return m

def _make_segment(p0: np.ndarray, p1: np.ndarray) -> "o3d.geometry.LineSet":
    pts = np.stack([p0.reshape(3), p1.reshape(3)], axis=0)
    lines = np.array([[0, 1]], dtype=np.int32)
    return o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )

def _make_cross(center: np.ndarray, size: float) -> "o3d.geometry.LineSet":
    c = center.reshape(3)
    s = float(size)
    pts = np.array([
        c + [-s, 0, 0], c + [ s, 0, 0],
        c + [0, -s, 0], c + [0,  s, 0],
        c + [0, 0, -s], c + [0, 0,  s],
    ], dtype=np.float64)
    lines = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
    return o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )

def _draw(geoms: List["o3d.geometry.Geometry"], title: str) -> None:
    o3d.visualization.draw_geometries(geoms, window_name=title)


# ------------------------ Paths ------------------------

def _this_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def _iter_dir_iter000() -> str:
    return os.path.join(_this_dir(), "sketch", "dsl_optimize", "optimize_iteration", "iter_000")

def _optimize_results_dir(iter_dir: str) -> str:
    return os.path.join(iter_dir, "optimize_results")

def _program_graph_json(iter_dir: str) -> str:
    return os.path.join(iter_dir, "program_graph.json")

def _fused_model_ply() -> str:
    return os.path.join(_this_dir(), "sketch", "3d_reconstruction", "fused_model.ply")

def _target_edit_dir() -> str:
    return os.path.join(_this_dir(), "sketch", "target_edit")


# ------------------------ Load candidates ------------------------

def _collect_candidate_labels(opt_results_dir: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(opt_results_dir):
        raise FileNotFoundError(f"Missing optimize_results_dir: {opt_results_dir}")

    for name in sorted(os.listdir(opt_results_dir)):
        label_dir = os.path.join(opt_results_dir, name)
        if not os.path.isdir(label_dir):
            continue
        after_json = os.path.join(label_dir, "bbox_after.json")
        if not os.path.isfile(after_json):
            continue
        rec = load_json(after_json)
        label = str(rec.get("label", name))
        out.append((label, after_json))
    return out

def _pick_random_non_unknown(cands: List[Tuple[str, str]]) -> Tuple[str, str]:
    good = [(lab, p) for (lab, p) in cands if not str(lab).startswith("unknown_")]
    if not good:
        raise RuntimeError("No valid labels found (all start with 'unknown_' or none exist).")
    return random.choice(good)


# ------------------------ Read bbox record + compute AABB ------------------------

def _read_bbox_after_record(bbox_after_path: str) -> Dict[str, Any]:
    rec = load_json(bbox_after_path)
    if not isinstance(rec, dict):
        raise RuntimeError(f"bbox_after.json is not a dict: {bbox_after_path}")
    return rec

def _get_world_aabb_from_record(rec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Prefer optimizer-provided opt_aabb_world; else compute from OBB.
    """
    opt = rec.get("opt_aabb_world", None)
    if isinstance(opt, dict) and "min" in opt and "max" in opt:
        mn = _np3(opt["min"])
        mx = _np3(opt["max"])
        meta = {"source": "opt_aabb_world"}
        return mn, mx, meta

    obb = rec.get("obb", {}) if isinstance(rec.get("obb", {}), dict) else {}
    center = _np3(obb.get("center", [0, 0, 0]))
    extent = _np3(obb.get("extent", [1, 1, 1]))
    R = _np33(obb.get("R", np.eye(3).tolist()))
    mn, mx = _obb_world_aabb(center, R, extent)
    meta = {"source": "obb_to_world_aabb"}
    return mn, mx, meta


# ------------------------ Drag-like edit: move ONE face ------------------------

def _drag_one_aabb_face(mn: np.ndarray, mx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Randomly pick one face among {minX,maxX,minY,maxY,minZ,maxZ} and move it.
    Move amount = [10%, 50%] of the box length along that axis.
    Direction randomly in/out, but clamped to keep mn < mx.
    """
    mn2 = mn.copy()
    mx2 = mx.copy()

    axis = random.randint(0, 2)  # 0=x,1=y,2=z
    side = random.choice(["min", "max"])

    length = float(max(1e-9, mx2[axis] - mn2[axis]))
    frac = random.uniform(0.10, 0.50)
    delta = frac * length

    # sign: +1 means increase coordinate, -1 decrease coordinate
    sign = random.choice([-1.0, 1.0])

    eps = 1e-6

    before_val = float(mn2[axis] if side == "min" else mx2[axis])

    if side == "min":
        # dragging min face
        new_val = before_val + sign * delta
        # clamp: must stay < max - eps
        new_val = min(new_val, float(mx2[axis] - eps))
        mn2[axis] = new_val
    else:
        # dragging max face
        new_val = before_val + sign * delta
        # clamp: must stay > min + eps
        new_val = max(new_val, float(mn2[axis] + eps))
        mx2[axis] = new_val

    meta = {
        "axis": int(axis),
        "axis_name": ["x", "y", "z"][axis],
        "side": side,
        "frac": float(frac),
        "delta": float(delta),
        "sign": float(sign),
        "before_face_value": float(before_val),
        "after_face_value": float(mn2[axis] if side == "min" else mx2[axis]),
        "aabb_before": {"min": mn.tolist(), "max": mx.tolist()},
        "aabb_after": {"min": mn2.tolist(), "max": mx2.tolist()},
    }
    return mn2, mx2, meta


# ------------------------ Graph relations (for vis 2) ------------------------

def _load_graph(graph_json: str) -> Dict[str, Any]:
    if not os.path.isfile(graph_json):
        raise FileNotFoundError(f"Missing program_graph.json: {graph_json} (run 12_graph_building.py first)")
    g = load_json(graph_json)
    if not isinstance(g, dict) or "nodes" not in g or "edges" not in g:
        raise RuntimeError("program_graph.json is not in expected format.")
    return g

def _graph_indices(graph: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[int, Dict[str, Any]]]:
    label_to_id: Dict[str, int] = {}
    id_to_node: Dict[int, Dict[str, Any]] = {}
    for n in graph.get("nodes", []):
        nid = int(n.get("id", -1))
        lab = str(n.get("label", ""))
        if nid >= 0 and lab:
            label_to_id[lab] = nid
            id_to_node[nid] = n
    return label_to_id, id_to_node

def _edges_for_node(graph: Dict[str, Any], nid: int) -> List[Dict[str, Any]]:
    out = []
    for e in graph.get("edges", []):
        if int(e.get("a", -1)) == nid or int(e.get("b", -1)) == nid:
            out.append(e)
    return out


# ------------------------ Visualization ------------------------

def _vis_edit_overlay_one_window(
    fused_ply: str,
    label: str,
    mn_before: np.ndarray,
    mx_before: np.ndarray,
    mn_after: np.ndarray,
    mx_after: np.ndarray,
) -> None:
    """
    One window: shape + BEFORE (blue) + AFTER (red stronger), using AABB.
    """
    fused = _load_fused_geometry_faint(fused_ply)

    ls_before = _make_aabb_lineset(mn_before, mx_before)
    ls_before.paint_uniform_color([0.0, 0.4, 1.0])  # blue

    ls_after = _make_aabb_lineset(mn_after, mx_after)
    ls_after.paint_uniform_color([1.0, 0.0, 0.0])   # red

    geoms: List[o3d.geometry.Geometry] = []
    geoms.extend(fused)

    geoms.append(ls_before)

    # stronger after: double draw
    geoms.append(ls_after)
    geoms.append(ls_after)

    _draw(geoms, title=f"EDIT OVERLAY (blue=before, red=after): {label}")

def _vis_relations_with_fixed_anchors(
    fused_ply: str,
    graph: Dict[str, Any],
    target_label: str,
    mn_before: np.ndarray,
    mx_before: np.ndarray,
    mn_after: np.ndarray,
    mx_after: np.ndarray,
) -> None:
    """
    One window:
    - fused shape faint
    - target BEFORE blue + AFTER red (strong)
    - all related neighbor bboxes from graph (unchanged)
    - anchor points/segments computed ONLY from BEFORE (graph) AABBs
      (i.e., use mn_before/mx_before for target; do NOT recompute with edited aabb)
    """
    fused = _load_fused_geometry_faint(fused_ply)

    label_to_id, id_to_node = _graph_indices(graph)
    if target_label not in label_to_id:
        raise RuntimeError(f"Target label '{target_label}' not found in program_graph.json nodes.")

    tid = int(label_to_id[target_label])
    edges_here = _edges_for_node(graph, tid)

    related_ids = {tid}
    for e in edges_here:
        related_ids.add(int(e.get("a", -1)))
        related_ids.add(int(e.get("b", -1)))

    # anchor size based on scene scale (median extent)
    ext_all = np.array([_np3(n.get("extent", [1, 1, 1])) for n in graph.get("nodes", [])], dtype=np.float64)
    med = float(np.median(ext_all)) if ext_all.size > 0 else 1.0
    pin_radius = float(max(1e-6, 0.01 * max(1e-12, med))) * 3.0
    cross_size = pin_radius * 1.5

    geoms: List[o3d.geometry.Geometry] = []
    geoms.extend(fused)

    # neighbor bboxes (unchanged; from graph's aabb_world)
    for rid in sorted(related_ids):
        if rid == tid:
            continue
        nn = id_to_node.get(rid)
        if nn is None:
            continue
        aabb = nn.get("aabb_world", {})
        mnB = _np3(aabb.get("min", [0, 0, 0]))
        mxB = _np3(aabb.get("max", [0, 0, 0]))
        ls = _make_aabb_lineset(mnB, mxB)
        ls.paint_uniform_color([0.25, 0.25, 0.25])  # strong gray
        geoms.append(ls)
        geoms.append(ls)

    # target boxes
    ls_before = _make_aabb_lineset(mn_before, mx_before)
    ls_before.paint_uniform_color([0.0, 0.4, 1.0])  # blue
    geoms.append(ls_before)

    ls_after = _make_aabb_lineset(mn_after, mx_after)
    ls_after.paint_uniform_color([1.0, 0.0, 0.0])   # red stronger
    geoms.append(ls_after)
    geoms.append(ls_after)

    # FIXED anchors: computed ONCE using BEFORE target AABB and neighbor AABB
    for e in edges_here:
        if str(e.get("type", "")) != "connected":
            continue

        a = int(e.get("a", -1))
        b = int(e.get("b", -1))
        other = b if a == tid else a
        nn = id_to_node.get(other)
        if nn is None:
            continue

        aabb = nn.get("aabb_world", {})
        mnB = _np3(aabb.get("min", [0, 0, 0]))
        mxB = _np3(aabb.get("max", [0, 0, 0]))

        pA, pB, pin = _aabb_closest_points(mn_before, mx_before, mnB, mxB)

        sph = _make_pin_sphere(pin, radius=pin_radius)
        sph.paint_uniform_color([1.0, 0.0, 1.0])  # magenta anchors
        geoms.append(sph)

        cross = _make_cross(pin, size=cross_size)
        cross.paint_uniform_color([1.0, 0.0, 1.0])
        geoms.append(cross)

        seg = _make_segment(pA, pB)
        seg.paint_uniform_color([1.0, 0.0, 1.0])
        geoms.append(seg)
        geoms.append(seg)

    _draw(geoms, title=f"RELATIONS + FIXED ANCHORS (blue=before, red=after): {target_label}")


# ------------------------ Save target_edit ------------------------

def _write_target_edit(
    out_dir: str,
    label: str,
    before_rec: Dict[str, Any],
    after_rec: Dict[str, Any],
    edit_meta: Dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_txt(os.path.join(out_dir, "target_label.txt"), f"{label}\n")
    save_json(os.path.join(out_dir, "bbox_before.json"), before_rec)
    save_json(os.path.join(out_dir, "bbox_after.json"), after_rec)
    save_json(os.path.join(out_dir, "edit_meta.json"), edit_meta)


# ------------------------ Main ------------------------

def main() -> None:
    random.seed()

    if o3d is None:
        print("[WARN] Open3D not available. Install open3d to visualize.")
        return

    iter_dir = _iter_dir_iter000()
    opt_dir = _optimize_results_dir(iter_dir)
    fused_ply = _fused_model_ply()
    out_dir = _target_edit_dir()
    graph_json = _program_graph_json(iter_dir)

    print("[EDIT] iter_dir :", iter_dir)
    print("[EDIT] opt_dir  :", opt_dir)
    print("[EDIT] graph    :", graph_json)
    print("[EDIT] fused_ply:", fused_ply)
    print("[EDIT] out_dir  :", out_dir)

    # pick target label
    cands = _collect_candidate_labels(opt_dir)
    label, bbox_after_path = _pick_random_non_unknown(cands)

    # load target bbox record (this is the "before")
    before_rec = _read_bbox_after_record(bbox_after_path)
    mn_before, mx_before, aabb_meta = _get_world_aabb_from_record(before_rec)

    # edit by dragging one face
    mn_after, mx_after, drag_meta = _drag_one_aabb_face(mn_before, mx_before)

    # after record: keep everything, but update opt_aabb_world to edited
    after_rec = _deepcopy_jsonable(before_rec)
    after_rec["opt_aabb_world"] = {"min": mn_after.tolist(), "max": mx_after.tolist()}
    # also store explicit edit field (handy for later AEP steps)
    after_rec["edit_aabb_world"] = {"min": mn_after.tolist(), "max": mx_after.tolist()}

    edit_meta = {
        "target_label": label,
        "source_bbox_after_json": os.path.abspath(bbox_after_path),
        "iter_dir": os.path.abspath(iter_dir),
        "aabb_source": aabb_meta,
        "edit": drag_meta,
        "note": "Drag ONE world-AABB face (placeholder for future command-driven AEP). Anchors are NOT recomputed.",
    }

    _write_target_edit(out_dir, label, before_rec, after_rec, edit_meta)

    print("[EDIT] picked label:", label)
    print("[EDIT] AABB before:", {"min": mn_before.tolist(), "max": mx_before.tolist()})
    print("[EDIT] AABB after :", {"min": mn_after.tolist(), "max": mx_after.tolist()})
    print("[EDIT] drag_meta  :", {k: drag_meta[k] for k in ["axis_name", "side", "frac", "sign", "delta"]})
    print("[EDIT] wrote target_edit:", out_dir)

    # vis 1: one window, before+after overlay
    _vis_edit_overlay_one_window(
        fused_ply=fused_ply,
        label=label,
        mn_before=mn_before,
        mx_before=mx_before,
        mn_after=mn_after,
        mx_after=mx_after,
    )

    # vis 2: relations + fixed anchors (computed from BEFORE only)
    graph = _load_graph(graph_json)
    _vis_relations_with_fixed_anchors(
        fused_ply=fused_ply,
        graph=graph,
        target_label=label,
        mn_before=mn_before,
        mx_before=mx_before,
        mn_after=mn_after,
        mx_after=mx_after,
    )


if __name__ == "__main__":
    main()
