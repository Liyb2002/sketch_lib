# AEP/same_pair_neighbor_edits.py
"""
AEP/same_pair_neighbor_edits.py

Method A (translation-only) for same_pair neighbors:
- Compute deltaA from AABB center shift of the edited (source) component.
- Map deltaA to neighbor deltaB using an axis-permutation induced by OBB rotations R
  (to handle cases where parts are "same" but oriented differently, e.g., X<->Y).

Then:
- mnB_new = mnB + deltaB
- mxB_new = mxB + deltaB

If R is unavailable or mapping is ambiguous, fall back to deltaB = deltaA.
"""

from typing import Any, Dict, Tuple
import numpy as np


def _np3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)


def _aabb_center(mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return 0.5 * (_np3(mn) + _np3(mx))


def _dominant_world_axis_per_local_axis(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given rotation matrix R (3x3) mapping local->world (world = R @ local),
    find for each local axis i (0..2) which world axis it mostly aligns with.

    Returns:
      dom_idx: (3,) int array, dom_idx[i] in {0,1,2} = argmax_k |R[k,i]|
      dom_sgn: (3,) float array, dom_sgn[i] = sign(R[dom_idx[i], i]) (unused for translation mapping)
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    dom_idx = np.zeros((3,), dtype=np.int64)
    dom_sgn = np.ones((3,), dtype=np.float64)
    for i in range(3):
        col = R[:, i]
        k = int(np.argmax(np.abs(col)))
        dom_idx[i] = k
        dom_sgn[i] = 1.0 if col[k] >= 0.0 else -1.0
    return dom_idx, dom_sgn


def _build_axis_mapping(A_dom: np.ndarray, B_dom: np.ndarray) -> Dict[int, int]:
    """
    Build a mapping from A local axis -> B local axis by matching dominant world axes.

    A_dom[i] = world axis index that A local axis i aligns with
    B_dom[j] = world axis index that B local axis j aligns with

    We match by equality of world axis indices.

    Returns:
      mapA2B: dict {i -> j}
    If ambiguous (duplicate world axes), mapping may be partial.
    """
    mapA2B: Dict[int, int] = {}
    used_B = set()
    for i in range(3):
        candidates = [j for j in range(3) if int(B_dom[j]) == int(A_dom[i]) and j not in used_B]
        if len(candidates) == 1:
            j = candidates[0]
            mapA2B[i] = j
            used_B.add(j)
        else:
            # ambiguous or missing; leave unmapped
            pass
    return mapA2B


def _map_delta_world_by_local_permutation(
    deltaA_world: np.ndarray,
    R_A: np.ndarray,
    R_B: np.ndarray,
) -> np.ndarray:
    """
    Map deltaA (world xyz) to deltaB (world xyz) by matching A/B local axes'
    dominant world axes.

    This handles axis swaps (e.g., A local x aligns to world X, but B local x aligns to world Y).

    If mapping is incomplete/ambiguous, fall back to deltaA_world.
    """
    deltaA_world = _np3(deltaA_world)

    A_dom, _ = _dominant_world_axis_per_local_axis(R_A)
    B_dom, _ = _dominant_world_axis_per_local_axis(R_B)

    # Map A local axis -> B local axis via same dominant world axis
    mapA2B = _build_axis_mapping(A_dom, B_dom)
    if len(mapA2B) < 3:
        # Too ambiguous; safe fallback
        return deltaA_world.copy()

    # Now convert mapping into world-axis permutation:
    # A local i corresponds to world axis A_dom[i]
    # B local j corresponds to world axis B_dom[j]
    # Since mapping pairs i->j share same dominant world axis, the world-axis component assignment is stable.
    # We simply keep delta component on that world axis.
    # (In practice this will often be identity; but if R encodes axis swap, it stays correct.)

    deltaB_world = np.zeros((3,), dtype=np.float64)
    # For each A local axis i, its component lives on world axis A_dom[i].
    # Since A_dom[i] == B_dom[j] for matched j, we place the same world-axis component.
    for i, j in mapA2B.items():
        w = int(A_dom[i])  # same as B_dom[j]
        deltaB_world[w] = deltaA_world[w]

    # For any axis left 0 because of numerical weirdness, keep original
    for k in range(3):
        if abs(deltaB_world[k]) < 1e-15 and abs(deltaA_world[k]) >= 1e-15:
            deltaB_world[k] = deltaA_world[k]

    return deltaB_world


def translate_same_pair_neighbor(
    mnA_before: np.ndarray,
    mxA_before: np.ndarray,
    mnA_after: np.ndarray,
    mxA_after: np.ndarray,
    mnB: np.ndarray,
    mxB: np.ndarray,
    nodeA: Dict[str, Any],
    nodeB: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      mnB_new, mxB_new, deltaB

    Uses deltaA = center(A_after)-center(A_before).
    Maps deltaA to deltaB using nodeA["R"] and nodeB["R"] if present; else deltaB=deltaA.
    """
    mnA_before = _np3(mnA_before)
    mxA_before = _np3(mxA_before)
    mnA_after  = _np3(mnA_after)
    mxA_after  = _np3(mxA_after)
    mnB = _np3(mnB)
    mxB = _np3(mxB)

    cA_before = _aabb_center(mnA_before, mxA_before)
    cA_after  = _aabb_center(mnA_after,  mxA_after)
    deltaA = cA_after - cA_before

    # Try orientation-aware mapping
    R_A = nodeA.get("R", None)
    R_B = nodeB.get("R", None)
    if R_A is not None and R_B is not None:
        try:
            deltaB = _map_delta_world_by_local_permutation(deltaA, np.asarray(R_A), np.asarray(R_B))
        except Exception:
            deltaB = deltaA.copy()
    else:
        deltaB = deltaA.copy()

    mnB_new = mnB + deltaB
    mxB_new = mxB + deltaB
    return mnB_new, mxB_new, deltaB
