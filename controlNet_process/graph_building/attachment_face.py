#!/usr/bin/env python3
# graph_building/attachment_face.py

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


# ------------------------------------------------------------
# Local frame helpers (object_space)
# ------------------------------------------------------------

def _world_to_object(p_world: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    # axes columns are object axes
    return (p_world - origin) @ axes


def _object_to_world(p_local: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return origin + axes @ p_local


# ------------------------------------------------------------
# OBB -> object-space AABB helpers
# ------------------------------------------------------------

def _get_obb_from_bbox_dict(bbox_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    bbox_entry is one item from label_bboxes_pca.json:
      bboxes_by_name[name] = {
        "obb_pca": {"center":[...], "axes":[[...]], "extents":[...]},
        ...
      }
    """
    if "obb_pca" not in bbox_entry:
        raise KeyError("bbox entry missing key 'obb_pca'")
    obb = bbox_entry["obb_pca"]
    if "center" not in obb or "extents" not in obb:
        raise KeyError("obb_pca missing 'center' or 'extents'")
    return obb


def _aabb_minmax_in_object_space(obb: Dict[str, Any], origin: np.ndarray, axes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Your PCA bboxes are aligned to object axes already.
    So in object coordinates, the box is axis-aligned:
      center_local = (center_world - origin) @ axes
      min = center_local - extents
      max = center_local + extents
    Return: (mn, mx, center_local)
    """
    c_world = np.array(obb["center"], dtype=np.float64)
    e = np.array(obb["extents"], dtype=np.float64)
    c_local = _world_to_object(c_world, origin, axes)
    mn = c_local - e
    mx = c_local + e
    return mn, mx, c_local


def _interval_overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return float(max(0.0, hi - lo))


def _axis_gap_and_faces(mn1, mx1, mn2, mx2, c1, c2, k: int) -> Tuple[float, Optional[str], Optional[str]]:
    """
    For axis k:
      - if separated: return positive gap, and deterministic face ids for each box
      - if overlapping: gap=0, faces=None (we resolve later if all axes overlap)
    Face id format: "+u0/-u0/+u1/-u1/+u2/-u2"
    """
    if mx1[k] < mn2[k]:
        gap = float(mn2[k] - mx1[k])
        # box1 touches with its + face, box2 touches with its - face (along k)
        return gap, f"+u{k}", f"-u{k}"
    if mx2[k] < mn1[k]:
        gap = float(mn1[k] - mx2[k])
        # box1 touches with its - face, box2 touches with its + face
        return gap, f"-u{k}", f"+u{k}"
    return 0.0, None, None


def _choose_contact_axis_and_faces(
    mn1: np.ndarray, mx1: np.ndarray, c1: np.ndarray,
    mn2: np.ndarray, mx2: np.ndarray, c2: np.ndarray,
    gap_tol: float,
    overlap_tol: float,
) -> Tuple[int, str, str, float, List[float]]:
    """
    Decide which axis is the "contact axis" and which faces are involved.

    Strategy:
    1) Compute per-axis gaps. If any gap>0 (boxes separated), pick the axis with the largest gap component
       (this is the axis along which the closest points are separated). Faces are from that axis.
    2) If gaps are all 0 (distance==0), boxes overlap/penetrate. Pick axis with *smallest overlap thickness*
       as the most plausible "contact axis". Faces determined by relative centers along that axis.

    Also returns overlap lengths on each axis for debug/printing.
    """
    overlaps = [
        _interval_overlap_len(mn1[0], mx1[0], mn2[0], mx2[0]),
        _interval_overlap_len(mn1[1], mx1[1], mn2[1], mx2[1]),
        _interval_overlap_len(mn1[2], mx1[2], mn2[2], mx2[2]),
    ]

    gaps = []
    faces = []
    for k in range(3):
        g, fa, fb = _axis_gap_and_faces(mn1, mx1, mn2, mx2, c1, c2, k)
        gaps.append(g)
        faces.append((fa, fb))

    max_gap = max(gaps)

    if max_gap > 0.0:
        # separated case: pick axis with largest gap component (dominant separation)
        k = int(np.argmax(np.array(gaps)))
        a_face, b_face = faces[k]
        # sanity
        if a_face is None or b_face is None:
            # should not happen if gap>0, but be safe
            # fallback based on centers:
            if c1[k] <= c2[k]:
                a_face, b_face = f"+u{k}", f"-u{k}"
            else:
                a_face, b_face = f"-u{k}", f"+u{k}"

        gap = float(gaps[k])

        # Optional strictness: require the other two axes to overlap enough
        other_axes = [ax for ax in [0, 1, 2] if ax != k]
        ok_other = True
        for ax in other_axes:
            if overlaps[ax] < overlap_tol:
                ok_other = False
                break
        # If not ok, still return but caller can decide to keep/ignore
        return k, a_face, b_face, gap, overlaps

    # overlap / distance==0 case: choose axis with smallest overlap thickness
    k = int(np.argmin(np.array(overlaps)))
    # determine facing by relative centers
    if c1[k] <= c2[k]:
        a_face, b_face = f"+u{k}", f"-u{k}"
    else:
        a_face, b_face = f"-u{k}", f"+u{k}"

    # treat as "gap" = 0 in overlap case
    return k, a_face, b_face, 0.0, overlaps


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def annotate_attachment_faces(
    bboxes_by_name: Dict[str, Any],
    attachments: List[Dict[str, Any]],
    object_space: Dict[str, Any],
    gap_tol: float,
    overlap_tol: float,
) -> List[Dict[str, Any]]:
    """
    Add face-level info to each attachment edge dict.

    Input attachments: list from find_attachment.find_attachments(), each like:
      {"a":..., "b":..., "distance":..., "anchor_local":..., "anchor_world":...}

    Output: same list, but each edge gets:
      "axis": 0/1/2
      "a_face": "+u0/-u0/+u1/-u1/+u2/-u2"
      "b_face": "+u0/-u0/+u1/-u1/+u2/-u2"
      "gap": float (dominant-axis gap, 0 if overlap case)
      "overlaps": [ov0, ov1, ov2]
    """
    origin = np.array(object_space["origin"], dtype=np.float64)
    axes = np.array(object_space["axes"], dtype=np.float64)
    if axes.shape != (3, 3):
        raise ValueError("object_space['axes'] must be 3x3")

    # precompute aabb per label in object space
    aabbs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, entry in bboxes_by_name.items():
        obb = _get_obb_from_bbox_dict(entry)
        mn, mx, c = _aabb_minmax_in_object_space(obb, origin, axes)
        aabbs[name] = (mn, mx, c)

    out = []
    for e in attachments:
        a = e["a"]
        b = e["b"]
        if a not in aabbs or b not in aabbs:
            out.append(e)
            continue

        mn1, mx1, c1 = aabbs[a]
        mn2, mx2, c2 = aabbs[b]

        axis, a_face, b_face, gap, overlaps = _choose_contact_axis_and_faces(
            mn1, mx1, c1, mn2, mx2, c2,
            gap_tol=float(gap_tol),
            overlap_tol=float(overlap_tol),
        )

        e2 = dict(e)
        e2["axis"] = int(axis)
        e2["a_face"] = a_face
        e2["b_face"] = b_face
        e2["gap"] = float(gap)
        e2["overlaps"] = [float(x) for x in overlaps]
        out.append(e2)

    return out


def print_attachment_faces(attachments_annotated: List[Dict[str, Any]]) -> None:
    """
    Convenience printer.
    """
    for e in attachments_annotated:
        a = e.get("a", "?")
        b = e.get("b", "?")
        a_face = e.get("a_face", None)
        b_face = e.get("b_face", None)
        axis = e.get("axis", None)
        dist = e.get("distance", None)
        gap = e.get("gap", None)
        overlaps = e.get("overlaps", None)

        if a_face is None or b_face is None:
            # print(f"[ATT_FACE][WARN] {a} <-> {b} : missing face info (keys not present)")
            continue

        # print(
        #     f"[ATT_FACE] {a}({a_face}) <-> {b}({b_face})  "
        #     f"axis=u{axis}  dist={dist:.6f}  gap={gap:.6f}  overlaps={overlaps}"
        # )
