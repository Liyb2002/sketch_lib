# graph_building/vis.py
#!/usr/bin/env python3
"""
graph_building/vis.py

Per-label visualization for program_graph.json:
- For each label, open TWO Open3D windows:
  1) WITH shape overlay (faint fused_model.ply)
  2) WITHOUT shape overlay (only bboxes + anchors)

Overlay shape path rule (your repo structure):
- fused_ply defaults to: <dir_of_12_graph_building.py>/sketch/3d_reconstruction/fused_model.ply
  (pass caller_file=__file__ from 12_graph_building.py)

Emphasis styling (as requested):
- fused shape: very faint light gray
- bboxes: strong (focus bbox pure red, neighbors strong blue), double-drawn to look thicker
- anchor pins: very strong (bigger magenta sphere + magenta crosshair + doubled segment)

No screenshots are saved. Interactive windows only.
"""

import os
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# ------------------------ small utils ------------------------

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _np3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)

def _np33(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3, 3)


# ------------------------ geometry helpers ------------------------

def _aabb_closest_points(
    mnA: np.ndarray, mxA: np.ndarray,
    mnB: np.ndarray, mxB: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute closest points pA (on A) and pB (on B) axis-wise:
    - if disjoint along axis: use opposing faces
    - if overlapping: use midpoint of overlap interval
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

def _make_pin_sphere(center: np.ndarray, radius: float) -> "o3d.geometry.TriangleMesh":
    m = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
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
    """
    3-axis crosshair centered at `center`.
    """
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

def _make_obb_lineset(center: np.ndarray, R: np.ndarray, extent: np.ndarray) -> "o3d.geometry.LineSet":
    obb = o3d.geometry.OrientedBoundingBox(center=center.reshape(3), R=R.reshape(3, 3), extent=extent.reshape(3))
    return o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)


# ------------------------ load fused geometry ------------------------

def _load_fused_geometry_faint(fused_ply: str) -> List["o3d.geometry.Geometry"]:
    """
    Robustly load fused_model.ply as point cloud or mesh.
    Make it faint for overlay.
    Prefer point cloud if non-empty.
    """
    if o3d is None:
        raise RuntimeError("Open3D not available. Install open3d to visualize.")
    if not os.path.isfile(fused_ply):
        raise FileNotFoundError(f"Missing fused model ply: {fused_ply}")

    # Prefer point cloud
    pcd = o3d.io.read_point_cloud(fused_ply)
    if pcd is not None and len(pcd.points) > 0:
        pcd.paint_uniform_color([0.75, 0.75, 0.75])  # faint light gray
        return [pcd]

    # Fallback to mesh
    mesh = o3d.io.read_triangle_mesh(fused_ply)
    if mesh is not None and len(mesh.vertices) > 0:
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.75, 0.75, 0.75])  # faint light gray
        return [mesh]

    raise RuntimeError(f"Loaded fused_model.ply but it is empty: {fused_ply}")


# ------------------------ Open3D drawing ------------------------

def _draw(geoms: List["o3d.geometry.Geometry"], title: str) -> None:
    o3d.visualization.draw_geometries(geoms, window_name=title)


# ------------------------ main API ------------------------

def run_graph_vis(
    iter_dir: str,
    graph_json: Optional[str] = None,
    fused_ply: Optional[str] = None,
    only_label: str = "",
    pin_radius_ratio: float = 0.01,
    max_labels: int = 0,
    show_non_connected_links: bool = False,
    caller_file: Optional[str] = None,
) -> None:
    """
    Visualize per label with interactive windows (NO saving).

    For each label: opens TWO windows
      1) "WITH_SHAPE"  : fused model + bboxes + anchors
      2) "NO_SHAPE"    : only bboxes + anchors

    Path rules (your repo structure):
    - fused_ply defaults to: <dir_of_12_graph_building.py>/sketch/3d_reconstruction/fused_model.ply
      -> pass caller_file=__file__ from 12_graph_building.py
    - graph_json defaults to <iter_dir>/program_graph.json
    """
    if o3d is None:
        print("[VIS] Open3D not available; skipping visualization.")
        return

    iter_dir = os.path.abspath(iter_dir)
    graph_json = graph_json or os.path.join(iter_dir, "program_graph.json")

    if caller_file is None:
        base_dir = os.getcwd()
        print("[VIS] WARN: caller_file not provided; using CWD for fused ply base:", base_dir)
    else:
        base_dir = os.path.dirname(os.path.abspath(caller_file))

    fused_ply = fused_ply or os.path.join(base_dir, "sketch", "3d_reconstruction", "fused_model.ply")

    if not os.path.isfile(graph_json):
        raise FileNotFoundError(f"Missing graph json: {graph_json}")

    graph: Dict[str, Any] = _load_json(graph_json)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes:
        raise RuntimeError("Graph has no nodes.")

    id_to_node = {int(n["id"]): n for n in nodes}
    label_to_id = {str(n.get("label", "")): int(n["id"]) for n in nodes}

    # Load faint fused geometry once (only used in the WITH_SHAPE window)
    fused_geoms_faint = _load_fused_geometry_faint(fused_ply)

    # Pin radius based on median extent (scene scale), then boost strongly
    ext = np.array([np.asarray(n.get("extent", [1, 1, 1]), dtype=np.float64).reshape(3) for n in nodes], dtype=np.float64)
    med = float(np.median(ext))
    pin_radius = float(max(1e-6, pin_radius_ratio * max(1e-12, med))) * 3.0  # stronger
    cross_size = float(pin_radius * 1.5)

    labels = [str(n.get("label", "")) for n in nodes]
    if only_label:
        if only_label not in label_to_id:
            raise ValueError(f"[VIS] only_label='{only_label}' not found.")
        labels = [only_label]

    if max_labels and max_labels > 0:
        labels = labels[: int(max_labels)]

    print("[VIS] graph_json:", graph_json)
    print("[VIS] iter_dir  :", iter_dir)
    print("[VIS] base_dir  :", base_dir)
    print("[VIS] fused_ply :", fused_ply)
    print("[VIS] labels    :", len(labels))
    print("[VIS] pin_radius:", pin_radius)

    def edges_for(node_id: int) -> List[Dict[str, Any]]:
        out = []
        for e in edges:
            if int(e.get("a", -1)) == node_id or int(e.get("b", -1)) == node_id:
                out.append(e)
        return out

    def build_geoms_for_label(label: str, include_shape: bool) -> List["o3d.geometry.Geometry"]:
        node_id = int(label_to_id[label])
        edges_here = edges_for(node_id)

        related_ids = {node_id}
        for e in edges_here:
            related_ids.add(int(e.get("a", -1)))
            related_ids.add(int(e.get("b", -1)))

        geoms: List[o3d.geometry.Geometry] = []
        if include_shape:
            geoms.extend(fused_geoms_faint)

        # Draw OBBs for current + related (double-draw to appear thicker)
        for rid in sorted(related_ids):
            nn = id_to_node.get(rid, None)
            if nn is None:
                continue
            c = _np3(nn.get("center", [0, 0, 0]))
            R = _np33(nn.get("R", np.eye(3)))
            extent = _np3(nn.get("extent", [0, 0, 0]))
            ls = _make_obb_lineset(c, R, extent)

            if rid == node_id:
                ls.paint_uniform_color([1.0, 0.0, 0.0])  # pure red focus
            else:
                ls.paint_uniform_color([0.0, 0.4, 1.0])  # strong blue neighbors

            geoms.append(ls)
            geoms.append(ls)  # visually "thicker"

        # Pins for connected edges + optional non-connected links
        for e in edges_here:
            et = str(e.get("type", ""))

            a = int(e.get("a", -1))
            b = int(e.get("b", -1))
            na = id_to_node.get(a, None)
            nb = id_to_node.get(b, None)
            if na is None or nb is None:
                continue

            if et == "connected":
                aabbA = na.get("aabb_world", {})
                aabbB = nb.get("aabb_world", {})
                mnA = _np3(aabbA.get("min", [0, 0, 0]))
                mxA = _np3(aabbA.get("max", [0, 0, 0]))
                mnB = _np3(aabbB.get("min", [0, 0, 0]))
                mxB = _np3(aabbB.get("max", [0, 0, 0]))

                pA, pB, pin = _aabb_closest_points(mnA, mxA, mnB, mxB)

                # BIG magenta sphere
                sph = _make_pin_sphere(pin, radius=pin_radius)
                sph.paint_uniform_color([1.0, 0.0, 1.0])
                geoms.append(sph)

                # Crosshair
                cross = _make_cross(pin, size=cross_size)
                cross.paint_uniform_color([1.0, 0.0, 1.0])
                geoms.append(cross)

                # Segment between closest points (double-draw)
                seg = _make_segment(pA, pB)
                seg.paint_uniform_color([1.0, 0.0, 1.0])
                geoms.append(seg)
                geoms.append(seg)

            elif show_non_connected_links:
                ca = _np3(na.get("center", [0, 0, 0]))
                cb = _np3(nb.get("center", [0, 0, 0]))
                seg = _make_segment(ca, cb)
                seg.paint_uniform_color([1.0, 1.0, 0.2])  # yellow-ish
                geoms.append(seg)
                geoms.append(seg)

        return geoms

    for label in labels:
        # Window 1: WITH shape overlay
        geoms_with = build_geoms_for_label(label, include_shape=True)
        _draw(geoms_with, title=f"graph_vis WITH_SHAPE: {label}")

        # Window 2: WITHOUT shape overlay
        geoms_no = build_geoms_for_label(label, include_shape=False)
        _draw(geoms_no, title=f"graph_vis NO_SHAPE: {label}")
