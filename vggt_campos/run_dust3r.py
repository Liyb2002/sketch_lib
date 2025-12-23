#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import trimesh

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_PATH = os.path.join(THIS_DIR, "dust3r")
if DUST3R_REPO_PATH not in sys.path:
    sys.path.append(DUST3R_REPO_PATH)

try:
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
except ImportError:
    print("Error: Could not import DUSt3R modules.")
    print(f"Make sure the 'dust3r' folder exists at: {DUST3R_REPO_PATH}")
    raise

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIEWS_DIR = os.path.join(THIS_DIR, "views")
OUTPUT_PLY = os.path.join(THIS_DIR, "final_scene_lines_only.ply")
CKPT_PATH = os.path.join(DUST3R_REPO_PATH, "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

LOAD_SIZE = 512

MASK_WHITE_BG = True
WHITE_THRESH = 0.97  # try 0.98/0.99 for cleaner paper background

PAIR_GRAPH = "complete"
BATCH_SIZE = 1
GLOBAL_NITER = 300
GLOBAL_LR = 0.01
GLOBAL_SCHEDULE = "cosine"

MAX_POINTS = 2_000_000  # None to disable

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def squeeze_batch(arr):
    """
    If arr is (1,...) squeeze to (...).
    Works for numpy arrays.
    """
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr

def img_to_hwc_np(img) -> np.ndarray:
    """
    Convert img (torch or np) into float32 numpy array (H,W,3) in [0,1].

    Handles:
      torch/np:
        - (1,3,H,W)  (batched CHW)
        - (3,H,W)
        - (1,H,W,3)  (batched HWC)
        - (H,W,3)
    """
    if torch.is_tensor(img):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)

    arr = arr.astype(np.float32)

    # squeeze batch dim if present
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array after squeeze, got shape {arr.shape}")

    # CHW -> HWC
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    # already HWC
    elif arr.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unrecognized RGB image shape: {arr.shape}")

    return arr

def make_white_mask_from_hwc(img_hwc: np.ndarray, thresh: float) -> np.ndarray:
    return (img_hwc > thresh).all(axis=2)  # (H,W) True=white bg

def apply_bg_mask_to_img_tensor(img_t: torch.Tensor, bg_mask_hw: np.ndarray) -> torch.Tensor:
    """
    Set background pixels to black for torch tensors.
    Supports:
      - (1,3,H,W)
      - (3,H,W)
      - (1,H,W,3)
      - (H,W,3)
    """
    if not torch.is_tensor(img_t):
        return img_t

    x = img_t.clone()

    # Convert bg mask to torch on same device
    bg = torch.from_numpy(bg_mask_hw).to(x.device)

    if x.ndim == 4 and x.shape[0] == 1:
        # (1,3,H,W) or (1,H,W,3)
        if x.shape[1] == 3:
            # NCHW
            x[0, :, bg] = 0.0
            return x
        if x.shape[-1] == 3:
            # NHWC
            x[0, bg, :] = 0.0
            return x

    if x.ndim == 3:
        if x.shape[0] == 3 and x.shape[-1] != 3:
            # CHW
            x[:, bg] = 0.0
            return x
        if x.shape[-1] == 3:
            # HWC
            x[bg, :] = 0.0
            return x

    # fallback: do nothing
    return x

def subsample_points(pts: np.ndarray, cols: np.ndarray, max_points: int):
    if max_points is None:
        return pts, cols
    n = pts.shape[0]
    if n <= max_points:
        return pts, cols
    idx = np.random.choice(n, size=max_points, replace=False)
    return pts[idx], cols[idx]

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("--- DUSt3R sketch reconstruction (lines-only) ---")
    print("Device:", DEVICE)

    if not os.path.exists(VIEWS_DIR):
        print("Error: Views folder not found:", VIEWS_DIR)
        return
    if not os.path.exists(CKPT_PATH):
        print("Error: Checkpoint not found:", CKPT_PATH)
        return

    print("Loading images from:", VIEWS_DIR)
    images = load_images(VIEWS_DIR, size=LOAD_SIZE)
    if len(images) == 0:
        print("No images found.")
        return
    print(f"Found {len(images)} images.")

    # Build BG masks from original images, then apply masking for inference
    bg_masks = []
    if MASK_WHITE_BG:
        print(f"Masking near-white background (thresh={WHITE_THRESH}) before inference...")
        for im in images:
            img_t = im["img"]
            img_hwc = img_to_hwc_np(img_t)  # now supports (1,3,H,W)
            bg = make_white_mask_from_hwc(img_hwc, WHITE_THRESH)
            bg_masks.append(bg)
            im["img"] = apply_bg_mask_to_img_tensor(img_t, bg)
    else:
        bg_masks = [None] * len(images)

    print("Loading DUSt3R model...")
    model = AsymmetricCroCo3DStereo.from_pretrained(CKPT_PATH).to(DEVICE)

    print("Running pairwise inference...")
    pairs = make_pairs(images, scene_graph=PAIR_GRAPH, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device=DEVICE, batch_size=BATCH_SIZE)

    print("Running global alignment...")
    scene = global_aligner(output, device=DEVICE, mode=GlobalAlignerMode.PointCloudOptimizer)
    _ = scene.compute_global_alignment(init="mst", niter=GLOBAL_NITER, schedule=GLOBAL_SCHEDULE, lr=GLOBAL_LR)

    print("Collecting points (filtering background pixels)...")
    pts3d_list = scene.get_pts3d()

    all_pts = []
    all_cols = []

    for i in range(len(images)):
        pts = pts3d_list[i].detach().cpu().numpy().reshape(-1, 3)

        img_hwc = img_to_hwc_np(images[i]["img"])   # (H,W,3)
        cols = img_hwc.reshape(-1, 3)

        valid = np.isfinite(pts).all(axis=1)

        if bg_masks[i] is not None:
            not_bg = ~bg_masks[i].reshape(-1)
        else:
            not_bg = np.linalg.norm(cols, axis=1) > 1e-6

        keep = valid & not_bg
        pts_k = pts[keep]
        cols_k = cols[keep]

        print(f"  view {i}: kept {pts_k.shape[0]} points")

        if pts_k.shape[0] > 0:
            all_pts.append(pts_k)
            all_cols.append(cols_k)

    if len(all_pts) == 0:
        print("Error: No valid points after filtering.")
        return

    final_pts = np.concatenate(all_pts, axis=0)
    final_cols = np.concatenate(all_cols, axis=0)
    final_cols = np.clip(final_cols, 0.0, 1.0)

    if MAX_POINTS is not None:
        before = final_pts.shape[0]
        final_pts, final_cols = subsample_points(final_pts, final_cols, MAX_POINTS)
        after = final_pts.shape[0]
        if after != before:
            print(f"Subsampled points: {before} -> {after}")

    print("Saving PLY to:", OUTPUT_PLY)
    pcd = trimesh.PointCloud(final_pts, colors=(final_cols * 255).astype(np.uint8))
    pcd.export(OUTPUT_PLY)

    print("Done.")
    print("Output:", OUTPUT_PLY)

if __name__ == "__main__":
    main()
