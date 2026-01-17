#!/usr/bin/env python3
# graph_building/vis.py

import re
import numpy as np
import open3d as o3d
from typing import Dict, Any, List, Tuple, Optional


# ----------------------------
# Helpers: bbox parsing / transforms
# ----------------------------

def _get_obb_pca(bbox_entry: Dict[str, Any]) -> Dict[str, Any]:
    if "obb_pca" not in bbox_entry:
        raise KeyError("bbox entry missing 'obb_pca'")
    obb = bbox_entry["obb_pca"]
    if "center" not in obb or "extents" not in obb:
        raise KeyError("obb_pca missing 'center' or 'extents'")
    return obb


def _world_to_object(p_world: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    # axes columns are object axes
    return (p_world - origin) @ axes


def _object_to_world(p_local: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    # axes columns are object axes
    return origin + axes @ p_local


def _aabb_minmax_local_from_obb(obb_pca: Dict[str, Any], origin: np.ndarray, axes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c_world = np.asarray(obb_pca["center"], dtype=np.float64)
    e = np.asarray(obb_pca["extents"], dtype=np.float64)
    c_local = _world_to_object(c_world, origin, axes)
    mn = c_local - e
    mx = c_local + e
    return mn, mx


def _bbox_lineset_world(
    mn_local: np.ndarray,
    mx_local: np.ndarray,
    origin: np.ndarray,
    axes: np.ndarray,
    color_rgb: Tuple[float, float, float],
) -> o3d.geometry.LineSet:
    x0, y0, z0 = mn_local.tolist()
    x1, y1, z1 = mx_local.tolist()

    corners_local = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float64)

    corners_world = np.array([_object_to_world(p, origin, axes) for p in corners_local], dtype=np.float64)

    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
    ], dtype=np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    colors = np.tile(np.array(color_rgb, dtype=np.float64)[None, :], (lines.shape[0], 1))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


# ----------------------------
# Helpers: attachment face mesh (FILLED)
# ----------------------------

_FACE_RE = re.compile(r"^([+-])u([012])$")

def _face_mesh_from_local_aabb(
    mn_local: np.ndarray,
    mx_local: np.ndarray,
    face_tag: str,
    origin: np.ndarray,
    axes: np.ndarray,
    color_rgb: Tuple[float, float, float],
) -> Optional[o3d.geometry.TriangleMesh]:
    """
    face_tag must be one of: +u0 -u0 +u1 -u1 +u2 -u2
    Creates a filled rectangle mesh for that face in WORLD space.
    """
    m = _FACE_RE.match(str(face_tag).strip())
    if m is None:
        return None

    sign = m.group(1)        # + or -
    ax = int(m.group(2))     # 0/1/2 in object frame
    fixed = mx_local[ax] if sign == "+" else mn_local[ax]

    if ax == 0:
        corners_local = np.array([
            [fixed, mn_local[1], mn_local[2]],
            [fixed, mx_local[1], mn_local[2]],
            [fixed, mx_local[1], mx_local[2]],
            [fixed, mn_local[1], mx_local[2]],
        ], dtype=np.float64)
    elif ax == 1:
        corners_local = np.array([
            [mn_local[0], fixed, mn_local[2]],
            [mx_local[0], fixed, mn_local[2]],
            [mx_local[0], fixed, mx_local[2]],
            [mn_local[0], fixed, mx_local[2]],
        ], dtype=np.float64)
    else:
        corners_local = np.array([
            [mn_local[0], mn_local[1], fixed],
            [mx_local[0], mn_local[1], fixed],
            [mx_local[0], mx_local[1], fixed],
            [mn_local[0], mx_local[1], fixed],
        ], dtype=np.float64)

    corners_world = np.array([_object_to_world(p, origin, axes) for p in corners_local], dtype=np.float64)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners_world)
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array(color_rgb, dtype=np.float64))
    return mesh


def _make_sphere(center_world: np.ndarray, r: float, color_rgb: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=float(r))
    s.translate(center_world.astype(np.float64))
    s.compute_vertex_normals()
    s.paint_uniform_color(np.array(color_rgb, dtype=np.float64))
    return s


# ----------------------------
# Robust relation parsing
# ----------------------------

def _sym_pairs_to_edges(symmetry: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs = symmetry.get("pairs", [])
    out: List[Tuple[str, str]] = []
    for item in pairs:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
        elif isinstance(item, dict) and "a" in item and "b" in item:
            out.append((str(item["a"]), str(item["b"])))
    return out


def _build_neighbor_map(
    names: List[str],
    symmetry: Dict[str, Any],
    attachments: List[Dict[str, Any]],
    containment: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, List[Any]]]:
    neigh = {n: {"sym": [], "att": [], "cont": []} for n in names}

    for a, b in _sym_pairs_to_edges(symmetry):
        if a in neigh and b in neigh:
            neigh[a]["sym"].append(b)
            neigh[b]["sym"].append(a)

    for e in attachments:
        a = str(e.get("a", ""))
        b = str(e.get("b", ""))
        if a in neigh and b in neigh:
            neigh[a]["att"].append(e)
            neigh[b]["att"].append(e)

    if containment is not None:
        for e in containment:
            a = str(e.get("a", ""))
            b = str(e.get("b", ""))
            if a in neigh and b in neigh:
                neigh[a]["cont"].append(e)
                neigh[b]["cont"].append(e)

    return neigh


# ----------------------------
# Main visualization: TWO windows per label
# ----------------------------

def verify_relations_vis(
    *,
    pts: np.ndarray,
    assigned_ids: np.ndarray,
    bboxes_by_name: Dict[str, Any],
    symmetry: Dict[str, Any],
    attachments: List[Dict[str, Any]],
    object_space: Dict[str, Any],
    containment: Optional[List[Dict[str, Any]]] = None,
    vis_anchor_points: bool = True,
    anchor_radius: float = 0.002,
    ignore_unknown: bool = False,
):
    """
    For each label:
      Window 1 (context):
        - point cloud overlay (gray)
        - focus bbox green
        - neighbor bboxes blue
        - NO attachment faces

      Window 2 (faces-only):
        - NO point cloud
        - focus bbox green
        - neighbor bboxes blue
        - attachment faces as FILLED BLUE rectangles
        - optional anchor points red
    """
    origin = np.asarray(object_space["origin"], dtype=np.float64)
    axes = np.asarray(object_space["axes"], dtype=np.float64)  # 3x3 columns
    if axes.shape != (3, 3):
        raise ValueError("object_space['axes'] must be 3x3")

    names = sorted(bboxes_by_name.keys())
    neigh = _build_neighbor_map(names, symmetry, attachments, containment)

    # base point cloud (gray)
    base_pcd = o3d.geometry.PointCloud()
    base_pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    base_pcd.paint_uniform_color([0.65, 0.65, 0.65])

    for focus in names:
        if ignore_unknown and focus.startswith("unknown_"):
            continue

        # collect neighbor names
        neighbor_names = set(neigh[focus]["sym"])

        for e in neigh[focus]["att"]:
            a = str(e.get("a", ""))
            b = str(e.get("b", ""))
            other = b if a == focus else a
            if other:
                neighbor_names.add(other)

        for e in neigh[focus]["cont"]:
            a = str(e.get("a", ""))
            b = str(e.get("b", ""))
            other = b if a == focus else a
            if other:
                neighbor_names.add(other)

        # focus bbox
        obb_focus = _get_obb_pca(bboxes_by_name[focus])
        mnF, mxF = _aabb_minmax_local_from_obb(obb_focus, origin, axes)
        focus_bbox = _bbox_lineset_world(mnF, mxF, origin, axes, color_rgb=(0.0, 1.0, 0.0))

        # neighbor bboxes
        neighbor_bboxes: List[o3d.geometry.Geometry] = []
        for nb in sorted(neighbor_names):
            if nb not in bboxes_by_name:
                continue
            if ignore_unknown and nb.startswith("unknown_"):
                continue
            obb_nb = _get_obb_pca(bboxes_by_name[nb])
            mnN, mxN = _aabb_minmax_local_from_obb(obb_nb, origin, axes)
            neighbor_bboxes.append(_bbox_lineset_world(mnN, mxN, origin, axes, color_rgb=(0.1, 0.4, 1.0)))

        # -----------------------------------------
        # Window 1: context (overlay + bboxes)
        # -----------------------------------------
        print("\n" + "=" * 80)
        print("[VIS] focus:", focus)
        print("[VIS] neighbors:", sorted(neighbor_names))
        ctx_geoms: List[o3d.geometry.Geometry] = [base_pcd, focus_bbox] + neighbor_bboxes
        o3d.visualization.draw_geometries(
            ctx_geoms,
            window_name=f"[CTX] {focus}: overlay+bbx (NO faces)",
        )

        # -----------------------------------------
        # Window 2: faces-only (no overlay)
        # -----------------------------------------
        face_geoms: List[o3d.geometry.Geometry] = [focus_bbox] + neighbor_bboxes

        if neigh[focus]["att"]:
            print("[VIS] attachments involving focus (faces):")
        else:
            print("[VIS] no attachment edges for this focus.")

        for e in neigh[focus]["att"]:
            a = str(e.get("a", ""))
            b = str(e.get("b", ""))
            other = b if a == focus else a

            a_face = e.get("a_face", None)
            b_face = e.get("b_face", None)
            axis = e.get("axis", None)
            gap = e.get("gap", None)

            print(f"  - {a} <-> {b} | other={other} | axis={axis} gap={gap} | a_face={a_face} b_face={b_face}")

            if vis_anchor_points and "anchor_world" in e:
                aw = np.asarray(e["anchor_world"], dtype=np.float64)
                face_geoms.append(_make_sphere(aw, anchor_radius, color_rgb=(1.0, 0.2, 0.2)))

            # draw BOTH faces (on A and B) as filled BLUE rectangles
            if a in bboxes_by_name and a_face is not None:
                obbA = _get_obb_pca(bboxes_by_name[a])
                mnA, mxA = _aabb_minmax_local_from_obb(obbA, origin, axes)
                meshA = _face_mesh_from_local_aabb(mnA, mxA, str(a_face), origin, axes, color_rgb=(0.0, 0.0, 1.0))
                if meshA is not None:
                    face_geoms.append(meshA)
                else:
                    print(f"    [WARN] cannot parse face tag for A: {a_face} (expected +u0/-u0/+u1/-u1/+u2/-u2)")

            if b in bboxes_by_name and b_face is not None:
                obbB = _get_obb_pca(bboxes_by_name[b])
                mnB, mxB = _aabb_minmax_local_from_obb(obbB, origin, axes)
                meshB = _face_mesh_from_local_aabb(mnB, mxB, str(b_face), origin, axes, color_rgb=(0.0, 0.0, 1.0))
                if meshB is not None:
                    face_geoms.append(meshB)
                else:
                    print(f"    [WARN] cannot parse face tag for B: {b_face} (expected +u0/-u0/+u1/-u1/+u2/-u2)")

        o3d.visualization.draw_geometries(
            face_geoms,
            window_name=f"[FACE] {focus}: bbx + FILLED attachment faces (NO overlay)",
        )
