#!/usr/bin/env python3
# compute_new_masks_from_aep.py
#
# Compute "new masks" by warping each label's original 2D mask using
# the 2D projective transform induced by the label's OBB before->after.
#
# Extra outputs added:
#   1) save 3D projected OBB overlay images per label:
#        sketch/back_project_masks/view_{x}/3d_project/{label}_obb_before_after.png
#   2) save per-view JSON containing warps:
#        sketch/back_project_masks/view_{x}/mask_warps.json
#
# NOTE (ONLY CHANGE MADE IN THIS EDIT):
# - In ALL mask visualizations:
#     * original mask is BLUE
#     * new mask is RED
#   (This affects: {label}_origmask_overlay.png and {label}_newmask_overlay.png.
#    Everything else remains unchanged.)

import os
import json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))

AEP_PATH   = os.path.join(ROOT, "sketch", "AEP", "aep_changes.json")
CONSTRAINTS_PATH = os.path.join(ROOT, "sketch", "AEP", "filtered_relations.json")
SCENE_DIR  = os.path.join(ROOT, "sketch", "3d_reconstruction")
VIEWS_DIR  = os.path.join(ROOT, "sketch", "views")
SEG_DIR    = os.path.join(ROOT, "sketch", "segmentation_original_image")
OUT_ROOT   = os.path.join(ROOT, "sketch", "back_project_masks")

NUM_VIEWS = 6

# Overlay style
ALPHA_FILL = 0.35
CONTOUR_THICK = 2

# OBB projection drawing style
OBB_THICK_BLUE = 3
OBB_THICK_RED  = 2


# ----------------------------
# Camera helpers (same convention as your 2D->3D mapping script)
# ----------------------------
def get_scaled_intrinsics(K_orig, src_w, src_h, target_w, target_h):
    scale_x = target_w / src_w
    scale_y = target_h / src_h
    K_new = K_orig.copy()
    K_new[0, 0] *= scale_x
    K_new[0, 2] *= scale_x
    K_new[1, 1] *= scale_y
    K_new[1, 2] *= scale_y
    return K_new


def project_world_to_px(P_world, w2c_4x4, K, H, W):
    """
    P_world: (N,3)
    Returns:
      px: (N,2) float
      valid: (N,) bool (z>0.01 and in image bounds)
    """
    N = P_world.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    Pw = np.hstack([P_world, ones])               # (N,4)
    Pc = (w2c_4x4 @ Pw.T).T                       # (N,4)

    x, y, z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid_z = z > 0.01
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    u[valid_z] = (x[valid_z] * fx / z[valid_z]) + cx
    v[valid_z] = (y[valid_z] * fy / z[valid_z]) + cy

    valid_px = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    px = np.stack([u, v], axis=1)
    return px, valid_px


# ----------------------------
# OBB geometry (PLACEMENT FIX)
# ----------------------------

def _get_object_space_T(object_space):
    origin = np.array(object_space["origin"], dtype=np.float64).reshape(3,)
    axes_list = np.array(object_space["axes"], dtype=np.float64)
    if axes_list.shape != (3, 3):
        raise ValueError("object_space['axes'] must be (3,3)")
    A_obj = axes_list.T  # rows u0,u1,u2 -> columns matrix
    return origin, A_obj


def _extract_anchor_pair_from_change(nb_change):
    dbg = nb_change.get("debug", {}) or {}
    edge = dbg.get("edge", {}) or {}
    aw = edge.get("anchor_world", None)
    if aw is None:
        return None

    op = nb_change.get("op", {}) or {}
    op_dbg = op.get("debug", {}) or {}
    vol_info = op_dbg.get("volume_info", {}) or {}
    al = vol_info.get("anchor_local", None)
    if al is None:
        return None

    return (np.array(al, dtype=np.float64).reshape(3,),
            np.array(aw, dtype=np.float64).reshape(3,))


def _decide_apply_obj2world(constraints, aep, eps_ok=1e-2):
    object_space = constraints.get("object_space", None)
    if object_space is None:
        return False, None, None

    origin, A_obj = _get_object_space_T(object_space)

    neigh = aep.get("neighbor_changes", {}) or {}
    best = None  # (err, name)
    for name, rec in neigh.items():
        if not isinstance(rec, dict):
            continue
        pair = _extract_anchor_pair_from_change(rec)
        if pair is None:
            continue
        anchor_local, anchor_world = pair
        pred_world = origin + (A_obj @ anchor_local)
        err = float(np.linalg.norm(pred_world - anchor_world))
        if best is None or err < best[0]:
            best = (err, name)

    if best is None:
        return False, None, None

    err, _ = best
    if err <= eps_ok:
        return True, origin, A_obj

    return False, None, None


def _obb_object_to_world(obb_obj, origin, A_obj):
    c_obj = np.asarray(obb_obj["center"], dtype=np.float64).reshape(3,)
    R_obj = np.asarray(obb_obj["axes"], dtype=np.float64)
    if R_obj.shape != (3, 3):
        raise ValueError("obb['axes'] must be (3,3)")

    c_world = origin + (A_obj @ c_obj)
    R_world = A_obj @ R_obj  # columns stay axes
    return {
        "center": c_world.tolist(),
        "axes": R_world.tolist(),
        "extents": obb_obj["extents"],
    }


def obb_corners_world(obb, apply_obj2world=False, origin=None, A_obj=None):
    if apply_obj2world:
        obb = _obb_object_to_world(obb, origin, A_obj)

    C = np.asarray(obb["center"], dtype=np.float64).reshape(3,)
    R = np.asarray(obb["axes"], dtype=np.float64)      # columns are axes
    E = np.asarray(obb["extents"], dtype=np.float64).reshape(3,)

    signs = np.array([
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ], dtype=np.float64)

    Q = signs * E[None, :]          # (8,3) local offsets
    return C[None, :] + (Q @ R.T)


def draw_projected_obb(img_bgr, corners_px, color_bgr, thickness=2):
    out = img_bgr.copy()
    p = corners_px.astype(np.int32)

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    for a, b in edges:
        cv2.line(out, tuple(p[a]), tuple(p[b]), color_bgr, thickness, lineType=cv2.LINE_AA)
    return out


# ----------------------------
# AEP change file parsing
# ----------------------------
def load_aep_items(aep_path):
    with open(aep_path, "r") as f:
        aep = json.load(f)

    target = aep.get("target", None)
    if not isinstance(target, str) or not target:
        raise ValueError("aep_changes.json missing 'target'")

    t_change = (aep.get("target_edit", {}) or {}).get("change", {}) or {}
    t_before = t_change.get("before_obb", None)
    t_after  = t_change.get("after_obb", None)
    if not (isinstance(t_before, dict) and isinstance(t_after, dict)):
        raise ValueError("aep_changes.json missing target_edit.change.before_obb/after_obb")

    items = [(target, t_before, t_after)]

    neigh = aep.get("neighbor_changes", {}) or {}
    if isinstance(neigh, dict):
        for name, rec in neigh.items():
            if not isinstance(rec, dict):
                continue
            b = rec.get("before_obb", None)
            a = rec.get("after_obb", None)
            if isinstance(name, str) and isinstance(b, dict) and isinstance(a, dict):
                items.append((name, b, a))

    return items


# ----------------------------
# Visual + warp utilities
# ----------------------------
def color_for_label(label: str):
    h = (abs(hash(label)) % 180)
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def overlay_mask_on_image(img_bgr, mask_u8, color_bgr, alpha=0.35, contour_thick=2):
    out = img_bgr.copy()
    if mask_u8 is None:
        return out

    binary = mask_u8 > 0
    if not np.any(binary):
        return out

    fill = np.zeros_like(out)
    fill[:] = color_bgr
    out[binary] = cv2.addWeighted(out[binary], 1.0 - alpha, fill[binary], alpha, 0)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color_bgr, contour_thick)

    return out


def compute_homography_from_obb(before_obb, after_obb, w2c, K, H, W,
                               apply_obj2world=False, origin=None, A_obj=None):
    Pw0 = obb_corners_world(before_obb, apply_obj2world=apply_obj2world, origin=origin, A_obj=A_obj)
    Pw1 = obb_corners_world(after_obb,  apply_obj2world=apply_obj2world, origin=origin, A_obj=A_obj)

    p0, v0 = project_world_to_px(Pw0, w2c, K, H, W)
    p1, v1 = project_world_to_px(Pw1, w2c, K, H, W)

    valid = v0 & v1
    idx = np.where(valid)[0]

    if idx.size < 4:
        return np.eye(3, dtype=np.float64), {
            "status": "identity_fallback",
            "valid_pairs": int(idx.size),
        }

    src = p0[idx].astype(np.float64)
    dst = p1[idx].astype(np.float64)

    Hmat, _ = cv2.findHomography(src, dst, method=0)
    if Hmat is None:
        return np.eye(3, dtype=np.float64), {
            "status": "findHomography_failed_identity",
            "valid_pairs": int(idx.size),
        }

    return Hmat.astype(np.float64), {
        "status": "ok",
        "valid_pairs": int(idx.size),
    }


def warp_mask(mask_u8, Hmat, out_h, out_w):
    warped = cv2.warpPerspective(
        mask_u8,
        Hmat,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    warped = (warped > 0).astype(np.uint8) * 255
    return warped


# ----------------------------
# Main
# ----------------------------
def main():
    if not os.path.exists(AEP_PATH):
        raise FileNotFoundError(f"Missing: {AEP_PATH}")

    # ---- placement fix: decide whether we must apply object_space ----
    with open(AEP_PATH, "r") as f:
        aep_raw = json.load(f)

    if os.path.exists(CONSTRAINTS_PATH):
        with open(CONSTRAINTS_PATH, "r") as f:
            constraints = json.load(f)
    else:
        constraints = {}

    apply_obj2world, origin, A_obj = _decide_apply_obj2world(constraints, aep_raw, eps_ok=1e-2)

    items = load_aep_items(AEP_PATH)
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Mask viz colors (BGR)
    ORIG_MASK_COLOR = (255, 0, 0)  # BLUE
    NEW_MASK_COLOR  = (0, 0, 255)  # RED

    for x in range(NUM_VIEWS):
        view_name = f"view_{x}"
        img_path = os.path.join(VIEWS_DIR, f"{view_name}.png")
        cam_path = os.path.join(SCENE_DIR, f"{view_name}_cam.json")
        seg_folder = os.path.join(SEG_DIR, view_name)

        if not os.path.exists(img_path):
            print(f"[skip] missing image: {img_path}")
            continue
        if not os.path.exists(cam_path):
            print(f"[skip] missing cam: {cam_path}")
            continue
        if not os.path.exists(seg_folder):
            print(f"[skip] missing seg folder: {seg_folder}")
            continue

        base = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if base is None:
            print(f"[skip] failed to read image: {img_path}")
            continue

        H_img, W_img = base.shape[:2]

        with open(cam_path, "r") as f:
            cam = json.load(f)

        w2c = np.array(cam["extrinsics_w2c"], dtype=np.float64)
        K_orig = np.array(cam["intrinsics"], dtype=np.float64)

        src_w = float(K_orig[0, 2] * 2.0)
        src_h = float(K_orig[1, 2] * 2.0)
        K = get_scaled_intrinsics(K_orig, src_w, src_h, W_img, H_img)

        out_dir = os.path.join(OUT_ROOT, view_name)
        os.makedirs(out_dir, exist_ok=True)

        proj_dir = os.path.join(out_dir, "3d_project")
        os.makedirs(proj_dir, exist_ok=True)

        all_new_overlay = base.copy()

        warp_json = {
            "view": view_name,
            "image": os.path.relpath(img_path, ROOT),
            "camera": os.path.relpath(cam_path, ROOT),
            "H": int(H_img),
            "W": int(W_img),
            "labels": {}
        }

        for (label, obb_before, obb_after) in items:
            # ---------- 3D projected overlay (always save) ----------
            Pw0 = obb_corners_world(obb_before, apply_obj2world=apply_obj2world, origin=origin, A_obj=A_obj)
            Pw1 = obb_corners_world(obb_after,  apply_obj2world=apply_obj2world, origin=origin, A_obj=A_obj)

            p0, _ = project_world_to_px(Pw0, w2c, K, H_img, W_img)
            p1, _ = project_world_to_px(Pw1, w2c, K, H_img, W_img)

            obb_vis = base.copy()
            # RED first, BLUE next so BLUE is on top
            obb_vis = draw_projected_obb(obb_vis, p1, (0, 0, 255), thickness=OBB_THICK_RED)    # BGR red (after)
            obb_vis = draw_projected_obb(obb_vis, p0, (255, 0, 0), thickness=OBB_THICK_BLUE)   # BGR blue (before)

            obb_vis_path = os.path.join(proj_dir, f"{label}_obb_before_after.png")
            cv2.imwrite(obb_vis_path, obb_vis)

            # ---------- mask warp (only if mask exists) ----------
            mask_path = os.path.join(seg_folder, f"{label}_mask.png")
            has_mask = os.path.exists(mask_path)

            label_entry = {
                "has_mask": bool(has_mask),
                "mask_path": os.path.relpath(mask_path, ROOT) if has_mask else None,
                "obb_before_after_overlay": os.path.relpath(obb_vis_path, ROOT),
                "homography_before_to_after": None,
                "homography_status": None,
                "valid_pairs": 0,
                "outputs": {}
            }

            if not has_mask:
                warp_json["labels"][label] = label_entry
                continue

            mask0 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask0 is None:
                warp_json["labels"][label] = label_entry
                continue
            if mask0.shape[:2] != (H_img, W_img):
                mask0 = cv2.resize(mask0, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

            Hmat, dbg = compute_homography_from_obb(
                obb_before, obb_after, w2c, K, H_img, W_img,
                apply_obj2world=apply_obj2world, origin=origin, A_obj=A_obj
            )
            mask1 = warp_mask(mask0, Hmat, H_img, W_img)

            out_mask_path = os.path.join(out_dir, f"{label}_mask_new.png")
            cv2.imwrite(out_mask_path, mask1)

            # ---- CHANGED: original mask always BLUE, new mask always RED ----
            orig_overlay = overlay_mask_on_image(
                base, mask0, ORIG_MASK_COLOR, alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK
            )
            new_overlay = overlay_mask_on_image(
                base, mask1, NEW_MASK_COLOR, alpha=ALPHA_FILL, contour_thick=CONTOUR_THICK
            )

            out_orig_overlay = os.path.join(out_dir, f"{label}_origmask_overlay.png")
            out_new_overlay  = os.path.join(out_dir, f"{label}_newmask_overlay.png")

            cv2.imwrite(out_orig_overlay, orig_overlay)
            cv2.imwrite(out_new_overlay, new_overlay)

            # keep aggregate overlay behavior unchanged (still per-label color)
            col = color_for_label(label)
            all_new_overlay = overlay_mask_on_image(all_new_overlay, mask1, col, alpha=ALPHA_FILL, contour_thick=1)

            label_entry["homography_before_to_after"] = Hmat.tolist()
            label_entry["homography_status"] = dbg.get("status")
            label_entry["valid_pairs"] = int(dbg.get("valid_pairs", 0))
            label_entry["outputs"] = {
                "mask_new": os.path.relpath(out_mask_path, ROOT),
                "origmask_overlay": os.path.relpath(out_orig_overlay, ROOT),
                "newmask_overlay": os.path.relpath(out_new_overlay, ROOT),
            }

            warp_json["labels"][label] = label_entry

            print(f"[ok] {view_name} {label}: mask warped | H={dbg['status']} pairs={dbg.get('valid_pairs')}")

        out_all = os.path.join(out_dir, "all_new_masks_overlay.png")
        cv2.imwrite(out_all, all_new_overlay)
        print(f"[ok] {view_name}: wrote {out_all}")

        warp_json["all_new_masks_overlay"] = os.path.relpath(out_all, ROOT)
        warp_json_path = os.path.join(out_dir, "mask_warps.json")
        with open(warp_json_path, "w") as f:
            json.dump(warp_json, f, indent=2)
        print(f"[ok] {view_name}: wrote {warp_json_path}")

    print("[done]")


if __name__ == "__main__":
    main()
