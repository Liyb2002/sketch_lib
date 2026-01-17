#!/usr/bin/env python3
"""
random_face_edit_and_vis.py

Edits a target OBB by moving ONE of its 6 faces (+u0/-u0/+u1/-u1/+u2/-u2)
by a signed delta in OBJECT-SPACE units. This makes the center move.

Pipeline assumptions (matches your saved JSON):
- sketch/AEP/initial_constraints.json contains:
  data["nodes"][label]["obb"] = {"center":[3], "axes":[[3],[3],[3]], "extents":[3]}
  where:
    - axes is 3x3, COLUMNS are world-space unit axes u0,u1,u2
    - extents are half-lengths along u0/u1/u2 (world units)

What this script does:
1) Choose target label (random non-unknown, or --target)
2) Choose face (random among 6, or --face)
3) Choose delta (random in [--delta-min, --delta-max], or --delta)
   delta > 0 expands outward, delta < 0 shrinks inward
4) Apply face move -> produces AFTER OBB (center changes, extent changes)
5) Save an edit JSON (explicit BEFORE + AFTER) to sketch/AEP/<out>.json
6) Visualize ONLY:
   - overlay shape (point cloud)
   - target PRE OBB
   - target POST OBB
   (no neighbors)

NOTE:
- delta is in object/world units consistent with your OBBs.
- We clamp to keep extents positive via --min-extent.
"""

import os
import json
import argparse
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import open3d as o3d


# -----------------------------
# IO helpers
# -----------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def safe_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# -----------------------------
# Selection helpers
# -----------------------------

def choose_target_label(nodes: Dict[str, Any], target: Optional[str]) -> str:
    if target is not None:
        if target not in nodes:
            raise ValueError(f"--target '{target}' not found in constraints nodes.")
        if target.startswith("unknown_"):
            raise ValueError(f"--target '{target}' is unknown_*. Please pick a non-unknown label.")
        return target

    candidates = [k for k in nodes.keys() if not k.startswith("unknown_")]
    if not candidates:
        raise ValueError("No non-unknown labels found in constraints nodes.")
    return random.choice(candidates)


def parse_face(face: str) -> Tuple[int, int]:
    """
    face string -> (axis_k, sign_s)
      "+u0" -> (0, +1)
      "-u2" -> (2, -1)
    """
    if len(face) != 3 or face[0] not in "+-" or face[1] != "u" or face[2] not in "012":
        raise ValueError("--face must be one of: +u0 -u0 +u1 -u1 +u2 -u2")
    s = +1 if face[0] == "+" else -1
    k = int(face[2])
    return k, s


def choose_face(face: Optional[str]) -> str:
    if face is not None:
        # validate
        parse_face(face)
        return face
    return random.choice(["+u0", "-u0", "+u1", "-u1", "+u2", "-u2"])


def choose_delta(delta: Optional[float], dmin: float, dmax: float) -> float:
    if delta is not None:
        return float(delta)
    if dmin > dmax:
        raise ValueError("--delta-min must be <= --delta-max")
    return float(random.uniform(dmin, dmax))


# -----------------------------
# OBB helpers
# -----------------------------

def get_obb(nodes: Dict[str, Any], name: str) -> Dict[str, Any]:
    obb = nodes[name].get("obb", None)
    if obb is None:
        raise ValueError(f"nodes['{name}']['obb'] is missing.")
    for k in ("center", "axes", "extents"):
        if k not in obb:
            raise ValueError(f"nodes['{name}']['obb'] missing '{k}'.")
    if len(obb["center"]) != 3 or len(obb["extents"]) != 3:
        raise ValueError(f"nodes['{name}']['obb'] center/extents must be length 3.")
    return obb


def apply_face_move(
    pre_obb: Dict[str, Any],
    face: str,
    delta: float,
    min_extent: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Move ONE face (+u0/-u0/+u1/-u1/+u2/-u2) by signed distance `delta`
    along that face's outward normal, keeping the opposite face FIXED.

    Conventions:
      - axes is 3x3, columns are world-space unit axes u0,u1,u2
      - extents are half-lengths along u0/u1/u2 (>=0)
      - face "+uk" means the MAX plane along axis k; "-uk" means the MIN plane.

    If moving "+uk" by delta:
      max' = max + delta,  min' = min
      => center' = center + (delta/2)*u_k
         extent' = extent + (delta/2)

    If moving "-uk" by delta (outward normal points toward -u_k):
      min' = min - delta,  max' = max
      => center' = center - (delta/2)*u_k
         extent' = extent + (delta/2)

    Notes:
      - delta > 0 expands outward, delta < 0 shrinks inward.
      - We clamp so the new half-extent on that axis is >= min_extent.
        This may reduce the applied delta magnitude.
    """
    k, s = parse_face(face)  # (axis index, sign), s=+1 for +face, s=-1 for -face

    c = np.array(pre_obb["center"], dtype=np.float64)
    R = np.array(pre_obb["axes"], dtype=np.float64)   # columns are axes
    e = np.array(pre_obb["extents"], dtype=np.float64)

    if R.shape != (3, 3):
        raise ValueError("obb['axes'] must be 3x3")
    if e.shape != (3,):
        raise ValueError("obb['extents'] must be length-3")

    old_e = float(e[k])
    u_k = R[:, k]  # world-space unit axis

    # For "move one face, keep opposite fixed":
    # extent changes by delta/2
    desired_new_e = old_e + 0.5 * float(delta)

    # Clamp to keep new extent >= min_extent
    if desired_new_e < float(min_extent):
        # old_e + 0.5*delta_applied = min_extent  => delta_applied = 2*(min_extent - old_e)
        delta_applied = 2.0 * (float(min_extent) - old_e)
    else:
        delta_applied = float(delta)

    new_e_k = old_e + 0.5 * delta_applied
    if new_e_k < float(min_extent) - 1e-12:
        # numerical safety
        new_e_k = float(min_extent)

    # Center shifts by (sign * delta/2) along u_k
    c2 = c + (s * 0.5 * delta_applied) * u_k

    new_e = e.copy()
    new_e[k] = new_e_k

    post_obb = {
        "center": c2.tolist(),
        "axes": pre_obb["axes"],       # orientation unchanged
        "extents": new_e.tolist(),
    }

    change_info = {
        "face": face,
        "axis": int(k),
        "axis_name": f"u{int(k)}",
        "sign": int(s),
        "delta_requested": float(delta),
        "delta_applied": float(delta_applied),
        "old_extent": float(old_e),
        "new_extent": float(new_e_k),
        "min_extent": float(min_extent),
    }

    return post_obb, change_info


def obb_to_lineset(obb: Dict[str, Any]) -> o3d.geometry.LineSet:
    center = np.array(obb["center"], dtype=np.float64)
    R = np.array(obb["axes"], dtype=np.float64)  # columns are axes
    extents_half = np.array(obb["extents"], dtype=np.float64)
    extent_full = 2.0 * extents_half

    o3d_obb = o3d.geometry.OrientedBoundingBox(center, R, extent_full)
    return o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_obb)


def paint_lineset(ls: o3d.geometry.LineSet, rgb) -> o3d.geometry.LineSet:
    colors = np.tile(np.array(rgb, dtype=np.float64)[None, :], (len(ls.lines), 1))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


# -----------------------------
# Main
# -----------------------------

def main():
    # ------------------------------------------------------------
    # HARD-CODED CONTROLS (edit these)
    # ------------------------------------------------------------

    # 1) target component (set to None to random-pick a non-unknown label)
    TARGET_LABEL = None          # e.g. "wheel_0"

    # 2) which face to move (set to None to random-pick among 6 faces)
    TARGET_FACE = None           # e.g. "+u2"  (one of: +u0 -u0 +u1 -u1 +u2 -u2)

    # 3) edit direction: expand or shrink
    #    If None: randomly choose expand/shrink each run.
    #    If +1: always expand outward; if -1: always shrink inward.
    EDIT_SIGN = None             # None / +1 / -1

    # 4) edit magnitude as a ratio of ORIGINAL half-extent on that axis
    #    Face displacement delta = ratio * original_extent(axis)
    #    (with corrected apply_face_move, this moves ONE face by delta and keeps the opposite face fixed)
    DELTA_RATIO_MIN = 0.3
    DELTA_RATIO_MAX = 0.5

    # Safety: minimum allowed half-extent after edit (prevents inversion)
    MIN_EXTENT = 1e-4

    # Random seed (set None to vary each run)
    SEED = 0

    # Toggle visualization
    DO_VIS = True

    # Output json name under sketch/AEP/
    OUT_FILENAME = "target_face_edit_change.json"

    # ------------------------------------------------------------
    # PATHS (usually no need to touch)
    # ------------------------------------------------------------
    repo_root = os.path.dirname(os.path.abspath(__file__))
    aep_dir = os.path.join(repo_root, "sketch", "AEP")
    constraints_path = os.path.join(aep_dir, "initial_constraints.json")
    overlay_ply = os.path.join(
        repo_root, "sketch", "partfield_overlay", "label_assignment_k20", "assignment_colored.ply"
    )
    out_path = os.path.join(aep_dir, OUT_FILENAME)

    if SEED is not None:
        random.seed(SEED)

    # ------------------------------------------------------------
    # LOAD CONSTRAINTS
    # ------------------------------------------------------------
    data = load_json(constraints_path)
    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict) or len(nodes) == 0:
        raise ValueError(f"No 'nodes' found in: {constraints_path}")

    # choose target and face (hardcoded or random)
    target = choose_target_label(nodes, TARGET_LABEL)
    face = choose_face(TARGET_FACE)

    # parse face -> axis index (k) + face sign (s)
    axis_k, face_sign = parse_face(face)

    # pick expand/shrink sign
    if EDIT_SIGN is None:
        mag_sign = random.choice([+1, -1])
    else:
        if EDIT_SIGN not in (+1, -1):
            raise ValueError("EDIT_SIGN must be None, +1, or -1.")
        mag_sign = int(EDIT_SIGN)

    # get pre obb and compute delta as ratio of original half-extent on that axis
    pre_obb = get_obb(nodes, target)
    e_k = float(pre_obb["extents"][axis_k])

    if DELTA_RATIO_MIN > DELTA_RATIO_MAX:
        raise ValueError("DELTA_RATIO_MIN must be <= DELTA_RATIO_MAX.")
    ratio = random.uniform(float(DELTA_RATIO_MIN), float(DELTA_RATIO_MAX))

    # delta is "face displacement" in world/object units
    delta = mag_sign * ratio * e_k

    # apply edit (one-face move, opposite face fixed)
    post_obb, change_info = apply_face_move(
        pre_obb=pre_obb,
        face=face,
        delta=float(delta),
        min_extent=float(MIN_EXTENT),
    )

    # ------------------------------------------------------------
    # SAVE EXPLICIT CHANGE (BEFORE + AFTER)
    # ------------------------------------------------------------
    payload = {
        "source_constraints": os.path.abspath(constraints_path),
        "target": target,
        "change": {
            "type": "move_single_face",
            "delta_ratio_min": float(DELTA_RATIO_MIN),
            "delta_ratio_max": float(DELTA_RATIO_MAX),
            "ratio_sampled": float(ratio),
            "edit_sign_setting": EDIT_SIGN,     # None/+1/-1 (hardcoded choice)
            "edit_sign_sampled": int(mag_sign), # actual sign used this run
            **change_info,
            "before_obb": pre_obb,
            "after_obb": post_obb,
        },
    }

    safe_write_json(out_path, payload)
    print("[EDIT] saved:", out_path)
    print(
        f"[EDIT] target={target} face={face} axis=u{axis_k} "
        f"ratio={ratio:.3f} sign={mag_sign:+d} "
        f"delta_req={change_info['delta_requested']:.6f} delta_applied={change_info['delta_applied']:.6f} "
        f"extent: {change_info['old_extent']:.6f}->{change_info['new_extent']:.6f}"
    )

    # ------------------------------------------------------------
    # VIS (ONLY overlay + pre/post target bbox)
    # ------------------------------------------------------------
    if not DO_VIS:
        return

    if not os.path.isfile(overlay_ply):
        raise FileNotFoundError(f"Overlay PLY not found: {overlay_ply}")

    pcd = o3d.io.read_point_cloud(overlay_ply)
    if pcd.has_points():
        gray = np.full((len(pcd.points), 3), 0.6, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(gray)

    ls_pre = paint_lineset(obb_to_lineset(pre_obb), (1.0, 0.0, 0.0))   # red
    ls_post = paint_lineset(obb_to_lineset(post_obb), (0.0, 1.0, 0.0)) # green

    o3d.visualization.draw_geometries(
        [pcd, ls_pre, ls_post],
        window_name=(
            f"Face Edit ONLY: {target} | pre=red post=green | "
            f"face={face} ratio={ratio:.2f} delta={change_info['delta_applied']:.4f}"
        ),
    )


if __name__ == "__main__":
    main()
