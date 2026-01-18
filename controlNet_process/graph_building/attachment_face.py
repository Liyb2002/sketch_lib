#!/usr/bin/env python3
# graph_building/attachment_face.py

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


VOL_Threshold = 0.3
# ------------------------------------------------------------
# Local frame helpers (object_space)
# ------------------------------------------------------------

def _world_to_object(p_world: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    # axes columns are object axes
    return (p_world - origin) @ axes


def _object_to_world(p_local: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return origin + axes @ p_local


# ------------------------------------------------------------
# OBB helpers
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
    if "center" not in obb or "axes" not in obb or "extents" not in obb:
        raise KeyError("obb_pca missing 'center' or 'axes' or 'extents'")
    return obb


def _obb_arrays(obb: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (center_world, axes_world, extents_half)
      - center_world: (3,)
      - axes_world:   (3,3) columns are axes in world
      - extents_half: (3,) half-lengths
    """
    c = np.array(obb["center"], dtype=np.float64).reshape(3)
    R = np.array(obb["axes"], dtype=np.float64)
    e = np.array(obb["extents"], dtype=np.float64).reshape(3)
    if R.shape != (3, 3):
        raise ValueError("obb['axes'] must be 3x3")
    return c, R, e


def _obb_volume(obb: Dict[str, Any]) -> float:
    # extents are half-lengths, so full side lengths = 2*e, volume = prod(2e) = 8*prod(e)
    _, _, e = _obb_arrays(obb)
    return float(8.0 * np.prod(e))


def _is_object_aligned_aabb(obb_axes_world: np.ndarray, object_axes_world: np.ndarray, tol: float = 1e-3) -> bool:
    """
    True if the OBB's axes are essentially the same as object_space axes (up to sign),
    meaning this bbox is an "object-space AABB lifted to world".

    We check M = object_axes^T * obb_axes should be ~ diagonal with entries +/-1.
    """
    A = np.array(object_axes_world, dtype=np.float64)
    B = np.array(obb_axes_world, dtype=np.float64)
    if A.shape != (3, 3) or B.shape != (3, 3):
        return False
    M = A.T @ B
    # absolute should be close to identity
    return bool(np.allclose(np.abs(M), np.eye(3), atol=tol, rtol=0.0))


def _aabb_minmax_in_object_space_from_obb(obb: Dict[str, Any], origin: np.ndarray, axes_obj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Only valid when obb is aligned to the object axes (AABB in object space).
      center_local = (center_world - origin) @ axes_obj
      min = center_local - extents
      max = center_local + extents
    Return: (mn, mx, center_local)
    """
    c_world, _, e = _obb_arrays(obb)
    c_local = _world_to_object(c_world, origin, axes_obj)
    mn = c_local - e
    mx = c_local + e
    return mn, mx, c_local


def _interval_overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return float(max(0.0, hi - lo))


def _axis_gap_and_faces(mn1, mx1, mn2, mx2, c1, c2, k: int) -> Tuple[float, Optional[str], Optional[str]]:
    """
    For axis k in object space (u0/u1/u2):
      - if separated: return positive gap, and deterministic face ids for each box
      - if overlapping: gap=0, faces=None
    Face id format: "+u0/-u0/+u1/-u1/+u2/-u2"
    """
    if mx1[k] < mn2[k]:
        gap = float(mn2[k] - mx1[k])
        return gap, f"+u{k}", f"-u{k}"
    if mx2[k] < mn1[k]:
        gap = float(mn1[k] - mx2[k])
        return gap, f"-u{k}", f"+u{k}"
    return 0.0, None, None


def _choose_contact_axis_and_faces(
    mn1: np.ndarray, mx1: np.ndarray, c1: np.ndarray,
    mn2: np.ndarray, mx2: np.ndarray, c2: np.ndarray,
    overlap_tol: float,
) -> Tuple[int, str, str, float, List[float], bool]:
    """
    Face-contact chooser (AABB-in-object-space only).

    Returns:
      axis, a_face, b_face, gap, overlaps, ok_other_axes

    - gap is >0 if separated along chosen axis, 0 if overlapping on that axis (which we treat as NOT face-contact)
    - ok_other_axes requires the other two axes to overlap by at least overlap_tol
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
    k = int(np.argmax(np.array(gaps)))  # dominant separation axis
    a_face, b_face = faces[k]

    if max_gap <= 0.0 or a_face is None or b_face is None:
        # overlapping everywhere (or degenerate): not a clean face contact
        # still return something deterministic for debug, but mark ok_other False and gap=0
        # faces from relative centers
        if c1[k] <= c2[k]:
            a_face, b_face = f"+u{k}", f"-u{k}"
        else:
            a_face, b_face = f"-u{k}", f"+u{k}"
        return k, a_face, b_face, 0.0, overlaps, False

    gap = float(gaps[k])

    other_axes = [ax for ax in [0, 1, 2] if ax != k]
    ok_other = True
    for ax in other_axes:
        if overlaps[ax] < float(overlap_tol):
            ok_other = False
            break

    return k, a_face, b_face, gap, overlaps, ok_other


def _estimate_overlap_volume_monte_carlo(
    obb_a: Dict[str, Any],
    obb_b: Dict[str, Any],
    n_samples: int = 2048,
    eps: float = 1e-9,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Approximate intersection volume of two OBBs by sampling points uniformly in the smaller OBB
    and testing inclusion in the other OBB.

    Returns:
      overlap_volume_est, frac_inside_small
    """
    ca, Ra, ea = _obb_arrays(obb_a)
    cb, Rb, eb = _obb_arrays(obb_b)

    vol_a = 8.0 * np.prod(ea)
    vol_b = 8.0 * np.prod(eb)

    # sample in smaller box for better efficiency
    if vol_a <= vol_b:
        cS, RS, eS = ca, Ra, ea
        cL, RL, eL = cb, Rb, eb
        vol_small = vol_a
    else:
        cS, RS, eS = cb, Rb, eb
        cL, RL, eL = ca, Ra, ea
        vol_small = vol_b

    rng = np.random.default_rng(seed)
    # uniform in [-e, e] per axis
    u = rng.uniform(low=-eS, high=eS, size=(int(n_samples), 3))
    p_world = cS[None, :] + (u @ RS.T)  # since RS columns are axes: world = c + RS @ u; with row vec use u @ RS.T

    # test inside larger OBB in its local coords
    v = (p_world - cL[None, :]) @ RL  # local coords in larger frame (row vec)
    inside = np.all(np.abs(v) <= (eL[None, :] + eps), axis=1)
    frac = float(np.mean(inside)) if p_world.shape[0] > 0 else 0.0
    overlap_vol = float(frac * vol_small)
    return overlap_vol, frac


# ------------------------------------------------------------
# Public API (same entry / same outputs, plus extra keys)
# ------------------------------------------------------------

def annotate_attachment_faces(
    bboxes_by_name: Dict[str, Any],
    attachments: List[Dict[str, Any]],
    object_space: Dict[str, Any],
    gap_tol: float,
    overlap_tol: float,
) -> List[Dict[str, Any]]:
    """
    Decide attachment relation type + (optionally) face info.

    User logic:
      1) If overlap_volume > 0.2 * min(volume_a, volume_b): volumetric attachment
      2) Else if a clean near-touching face exists AND both bboxes are object-aligned AABB: face attachment
      3) Else: point attachment

    Output format remains compatible:
      - Existing keys "axis","a_face","b_face","gap","overlaps" are only added for FACE attachments.
      - We add extra debug keys (safe for downstream that ignores unknown fields):
          "relation_type": "volume"|"face"|"point"
          "overlap_volume_est", "overlap_frac_small", "vol_a", "vol_b"
          "aabb_aligned_a", "aabb_aligned_b"
    """
    origin = np.array(object_space["origin"], dtype=np.float64)
    axes_obj = np.array(object_space["axes"], dtype=np.float64)
    if axes_obj.shape != (3, 3):
        raise ValueError("object_space['axes'] must be 3x3")

    # cache OBBs
    obbs: Dict[str, Dict[str, Any]] = {}
    for name, entry in bboxes_by_name.items():
        try:
            obbs[name] = _get_obb_from_bbox_dict(entry)
        except Exception:
            continue

    out: List[Dict[str, Any]] = []
    for e in attachments:
        a = e.get("a", None)
        b = e.get("b", None)
        if a is None or b is None or a not in obbs or b not in obbs:
            out.append(e)
            continue

        obb_a = obbs[a]
        obb_b = obbs[b]

        # volumes
        vol_a = _obb_volume(obb_a)
        vol_b = _obb_volume(obb_b)
        vol_min = min(vol_a, vol_b)

        # (1) volumetric attachment check (robust to OBB)
        overlap_vol, overlap_frac = _estimate_overlap_volume_monte_carlo(
            obb_a, obb_b,
            n_samples=2048,
            seed=0,
        )
        is_volume = overlap_vol > (VOL_Threshold * vol_min)

        # (2) face attachment check only if BOTH are object-aligned AABB
        ca, Ra, ea = _obb_arrays(obb_a)
        cb, Rb, eb = _obb_arrays(obb_b)
        a_aligned = _is_object_aligned_aabb(Ra, axes_obj, tol=1e-3)
        b_aligned = _is_object_aligned_aabb(Rb, axes_obj, tol=1e-3)

        e2 = dict(e)
        e2["vol_a"] = float(vol_a)
        e2["vol_b"] = float(vol_b)
        e2["overlap_volume_est"] = float(overlap_vol)
        e2["overlap_frac_small"] = float(overlap_frac)
        e2["aabb_aligned_a"] = bool(a_aligned)
        e2["aabb_aligned_b"] = bool(b_aligned)

        if is_volume:
            e2["relation_type"] = "volume"
            out.append(e2)
            continue

        if a_aligned and b_aligned:
            mn1, mx1, c1 = _aabb_minmax_in_object_space_from_obb(obb_a, origin, axes_obj)
            mn2, mx2, c2 = _aabb_minmax_in_object_space_from_obb(obb_b, origin, axes_obj)

            axis, a_face, b_face, gap, overlaps, ok_other = _choose_contact_axis_and_faces(
                mn1, mx1, c1,
                mn2, mx2, c2,
                overlap_tol=float(overlap_tol),
            )

            # "very close": gap must be small AND other axes overlap enough
            # also require strictly separated along that axis (gap>0), otherwise it's overlap which we treat as not face-contact
            if (gap > 0.0) and (gap <= float(gap_tol)) and ok_other:
                e2["relation_type"] = "face"
                e2["axis"] = int(axis)
                e2["a_face"] = a_face
                e2["b_face"] = b_face
                e2["gap"] = float(gap)
                e2["overlaps"] = [float(x) for x in overlaps]
                out.append(e2)
                continue

        # (3) point attachment fallback
        e2["relation_type"] = "point"
        out.append(e2)

    return out


def print_attachment_faces(attachments_annotated: List[Dict[str, Any]]) -> None:
    """
    Convenience printer.
    Prints only face attachments (same behavior as before: skips ones without faces).
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
            continue

        # Uncomment if you want printing:
        # print(
        #     f"[ATT_FACE] {a}({a_face}) <-> {b}({b_face})  "
        #     f"axis=u{axis}  dist={dist:.6f}  gap={gap:.6f}  overlaps={overlaps}"
        # )
