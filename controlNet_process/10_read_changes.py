#!/usr/bin/env python3
"""
random_face_edit_save_only_with_vis.py

Randomly (or hardcoded) pick ONE non-unknown component, pick ONE face (+u0/-u0/+u1/-u1/+u2/-u2),
apply a one-face move (opposite face fixed), SAVE ONLY the changed bbox (before/after),
AND visualize ONLY:
  - overlay shape (point cloud)
  - target PRE OBB
  - target POST OBB
(no neighbors)

Writes: sketch/AEP/target_face_edit_change.json
"""

import os
import json
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
            raise ValueError(f"TARGET_LABEL '{target}' not found in constraints nodes.")
        if target.startswith("unknown_"):
            raise ValueError(f"TARGET_LABEL '{target}' is unknown_*. Please pick a non-unknown label.")
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
        raise ValueError("face must be one of: +u0 -u0 +u1 -u1 +u2 -u2")
    s = +1 if face[0] == "+" else -1
    k = int(face[2])
    return k, s


def choose_face(face: Optional[str]) -> str:
    if face is not None:
        parse_face(face)
        return face
    return random.choice(["+u0", "-u0", "+u1", "-u1", "+u2", "-u2"])


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
    Move ONE face by signed distance `delta` along its outward normal,
    keeping the opposite face FIXED.

    With half-extent e and center c along axis k:
      extent' = e + delta/2
      center' = c + (sign * delta/2) * u_k

    Clamp so new half-extent >= min_extent.
    """
    k, s = parse_face(face)

    c = np.array(pre_obb["center"], dtype=np.float64)
    R = np.array(pre_obb["axes"], dtype=np.float64)   # columns are axes
    e = np.array(pre_obb["extents"], dtype=np.float64)

    if R.shape != (3, 3):
        raise ValueError("obb['axes'] must be 3x3")
    if e.shape != (3,):
        raise ValueError("obb['extents'] must be length-3")

    old_e = float(e[k])
    u_k = R[:, k]

    desired_new_e = old_e + 0.5 * float(delta)

    if desired_new_e < float(min_extent):
        delta_applied = 2.0 * (float(min_extent) - old_e)
    else:
        delta_applied = float(delta)

    new_e_k = old_e + 0.5 * delta_applied
    if new_e_k < float(min_extent) - 1e-12:
        new_e_k = float(min_extent)

    c2 = c + (s * 0.5 * delta_applied) * u_k

    new_e = e.copy()
    new_e[k] = new_e_k

    post_obb = {
        "center": c2.tolist(),
        "axes": pre_obb["axes"],
        "extents": new_e.tolist(),
    }

    info = {
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
    return post_obb, info


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
# Main (hardcoded controls)
# -----------------------------

def main():
    # ------------------------------------------------------------
    # HARD-CODED CONTROLS (edit these)
    # ------------------------------------------------------------

    # target label (None => random non-unknown)
    TARGET_LABEL = None        # e.g. "wheel_0"

    # face (None => random among 6)
    TARGET_FACE = None         # e.g. "+u2"

    # delta is a RATIO of the ORIGINAL half-extent on that axis
    # face displacement magnitude = ratio * original_extent(axis)
    DELTA_RATIO_MIN = 0.3
    DELTA_RATIO_MAX = 0.5

    # randomly choose increase/decrease each run:
    #   +1 => expand outward
    #   -1 => shrink inward
    RANDOM_EXPAND_SHRINK = True

    # minimum half-extent allowed after edit (prevents inversion)
    MIN_EXTENT = 1e-4

    # seed (None => vary)
    SEED = 0

    OUT_FILENAME = "target_face_edit_change.json"
    DO_VIS = True

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    repo_root = os.path.dirname(os.path.abspath(__file__))
    aep_dir = os.path.join(repo_root, "sketch", "AEP")
    constraints_path = os.path.join(aep_dir, "initial_constraints.json")
    out_path = os.path.join(aep_dir, OUT_FILENAME)

    overlay_ply = os.path.join(
        repo_root, "sketch", "partfield_overlay", "label_assignment_k20", "assignment_colored.ply"
    )

    if SEED is not None:
        random.seed(SEED)

    # ------------------------------------------------------------
    # Load constraints + pick target/face
    # ------------------------------------------------------------
    data = load_json(constraints_path)
    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict) or len(nodes) == 0:
        raise ValueError(f"No 'nodes' found in: {constraints_path}")

    target = choose_target_label(nodes, TARGET_LABEL)
    face = choose_face(TARGET_FACE)
    axis_k, _face_sign = parse_face(face)   # which axis (0/1/2) this face is on

    pre_obb = get_obb(nodes, target)
    e_k = float(pre_obb["extents"][axis_k])  # original half-extent

    # ------------------------------------------------------------
    # Choose delta = (+/-) ratio * original_extent(axis)
    # ------------------------------------------------------------
    if DELTA_RATIO_MIN > DELTA_RATIO_MAX:
        raise ValueError("DELTA_RATIO_MIN must be <= DELTA_RATIO_MAX")
    ratio = random.uniform(float(DELTA_RATIO_MIN), float(DELTA_RATIO_MAX))

    if RANDOM_EXPAND_SHRINK:
        mag_sign = random.choice([+1, -1])
    else:
        mag_sign = +1  # default expand if you disable randomness

    delta = float(mag_sign) * float(ratio) * e_k  # face displacement in world/object units

    # apply edit (one-face move, opposite face fixed)
    post_obb, info = apply_face_move(
        pre_obb=pre_obb,
        face=face,
        delta=delta,
        min_extent=float(MIN_EXTENT),
    )

    # ------------------------------------------------------------
    # SAVE ONLY the changed bbox (before/after)
    # ------------------------------------------------------------
    payload = {
        "target": target,
        "change": {
            "type": "move_single_face",
            "delta_ratio_min": float(DELTA_RATIO_MIN),
            "delta_ratio_max": float(DELTA_RATIO_MAX),
            "ratio_sampled": float(ratio),
            "expand_or_shrink": "expand" if mag_sign > 0 else "shrink",
            **info,
            "before_obb": pre_obb,
            "after_obb": post_obb,
        },
    }
    safe_write_json(out_path, payload)
    print("[EDIT] saved:", out_path)
    print(
        f"[EDIT] target={target} face={face} axis=u{axis_k} "
        f"ratio={ratio:.3f} ({'+' if mag_sign > 0 else '-'}) "
        f"delta_req={info['delta_requested']:.6f} delta_applied={info['delta_applied']:.6f} "
        f"extent: {info['old_extent']:.6f}->{info['new_extent']:.6f}"
    )

    # ------------------------------------------------------------
    # VIS: overlay shape + ONLY pre/post bbox
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
            f"Target ONLY: {target} pre=red post=green "
            f"face={face} ratio={ratio:.2f} delta={info['delta_applied']:.4f}"
        ),
    )


if __name__ == "__main__":
    main()
