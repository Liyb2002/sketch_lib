#!/usr/bin/env python3
"""
pc_to_lowpoly_mesh.py

Goal: noisy point cloud (model.ply) -> low-poly mesh, avoiding the "giant blob" problem
caused by scattered points bridging the reconstruction.

Input:  model.ply (point cloud) in the same folder as this script
Output (same folder):
  - cleaned_points.ply
  - mesh_lowpoly.ply
  - mesh_lowpoly.glb  (best-effort; if GLB export fails, PLY still saved)

Core idea:
1) Quick, aggressive spatial filter FIRST:
   - coarse voxelization + connected component on occupied voxels
   - keep only largest (or top-K) components
   This removes scattered points "everywhere" that otherwise create a huge chunk mesh.

2) Then normal denoise + downsample + normals

3) Mesh: Alpha-shape -> BPA -> Poisson(trim) fallback

4) Keep largest mesh component + decimate to target tris
"""

import os
import sys
from collections import deque

import numpy as np
import open3d as o3d


# ----------------------------
# Basic helpers
# ----------------------------

def bbox_diag(pcd: o3d.geometry.PointCloud) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(aabb.get_extent(), dtype=np.float64)
    return float(np.linalg.norm(extent))


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def keep_largest_mesh_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    tri_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if cluster_n_triangles.size == 0:
        return mesh
    largest = int(cluster_n_triangles.argmax())
    remove_mask = np.asarray(tri_clusters) != largest
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


# ----------------------------
# Key "quick filter": voxel connectivity component pruning
# ----------------------------

def keep_largest_voxel_component(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    keep_top_k: int = 1,
) -> o3d.geometry.PointCloud:
    """
    Voxelize points (integer voxel coords), build 6-neighborhood connectivity between occupied voxels,
    keep points that belong to the largest (or top-K) connected voxel components.

    This is robust against scattered junk points that would otherwise cause alpha/poisson to bridge
    into one big blob.
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd

    # Point -> voxel coord
    v = np.floor(pts / voxel_size).astype(np.int64)

    # Unique voxels and inverse mapping (each point -> voxel index in `vox`)
    vox, inv = np.unique(v, axis=0, return_inverse=True)

    # coord -> index
    voxel_to_idx = {tuple(vc): i for i, vc in enumerate(vox)}

    # 6-neighborhood offsets
    offsets = np.array([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1], [ 0, 0,-1],
    ], dtype=np.int64)

    # adjacency list over voxel indices
    neighbors = [[] for _ in range(len(vox))]
    for i, vc in enumerate(vox):
        for off in offsets:
            nb = tuple((vc + off).tolist())
            j = voxel_to_idx.get(nb)
            if j is not None:
                neighbors[i].append(j)

    # connected components via BFS
    visited = np.zeros(len(vox), dtype=bool)
    comps = []
    for i in range(len(vox)):
        if visited[i]:
            continue
        q = deque([i])
        visited[i] = True
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in neighbors[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    q.append(nxt)
        comps.append(comp)

    # Sort components by how many points they contain (not just voxel count)
    voxel_point_counts = np.bincount(inv, minlength=len(vox))
    comps_sorted = sorted(
        comps,
        key=lambda c: int(voxel_point_counts[np.asarray(c, dtype=np.int64)].sum()),
        reverse=True
    )

    keep_top_k = max(1, int(keep_top_k))
    keep_voxel_ids = set()
    for c in comps_sorted[:keep_top_k]:
        keep_voxel_ids.update(c)

    keep_mask = np.isin(inv, np.fromiter(keep_voxel_ids, dtype=np.int64))
    idx = np.where(keep_mask)[0]
    return pcd.select_by_index(idx)


# ----------------------------
# Meshing methods
# ----------------------------

def mesh_alpha_shape(pcd: o3d.geometry.PointCloud, alpha: float) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)


def mesh_ball_pivoting(pcd: o3d.geometry.PointCloud, radii):
    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(list(radii))
    )


def mesh_poisson_trim(pcd: o3d.geometry.PointCloud, depth: int, trim_quantile: float):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    if densities.size > 0:
        thr = np.quantile(densities, trim_quantile)
        keep = densities >= thr
        mesh.remove_vertices_by_mask(~keep)
    return mesh


def try_build_mesh(pcd: o3d.geometry.PointCloud, final_voxel: float):
    """
    For low-poly after filtering, try:
      1) Alpha shape (tight-ish)
      2) BPA
      3) Poisson+trim fallback
    """
    # --- Alpha shape: scale alpha with voxel
    for mult in (2.5, 3.5, 5.0, 7.0):
        alpha = final_voxel * mult
        try:
            m = mesh_alpha_shape(pcd, alpha)
            m = clean_mesh(m)
            if len(m.triangles) > 200:
                return m, f"alpha_shape(alpha={alpha:.6g})"
        except Exception:
            pass

    # --- BPA: scale radii with voxel
    try:
        radii = [final_voxel * 1.5, final_voxel * 2.5, final_voxel * 3.5]
        m = mesh_ball_pivoting(pcd, radii)
        m = clean_mesh(m)
        if len(m.triangles) > 200:
            return m, f"ball_pivoting(radii={[float(r) for r in radii]})"
    except Exception:
        pass

    # --- Poisson + trim (watertight-ish fallback)
    try:
        m = mesh_poisson_trim(pcd, depth=8, trim_quantile=0.10)
        m = clean_mesh(m)
        if len(m.triangles) > 200:
            return m, "poisson(depth=8, trim=0.10)"
    except Exception:
        pass

    raise RuntimeError("All meshing methods failed (alpha/BPA/poisson).")


# ----------------------------
# Main
# ----------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(here, "model.ply")
    if not os.path.exists(in_path):
        print(f"[ERROR] Not found: {in_path}")
        sys.exit(1)

    print(f"[LOAD] {in_path}")
    pcd = o3d.io.read_point_cloud(in_path)
    if pcd.is_empty():
        print("[ERROR] Loaded point cloud is empty.")
        sys.exit(1)

    diag = bbox_diag(pcd)
    if not np.isfinite(diag) or diag <= 0:
        print("[ERROR] Bad bbox diagonal. Is the point cloud valid?")
        sys.exit(1)

    # ---- Parameters (auto from bbox diag) ----
    # Coarse voxel for connectivity pruning: aggressive removal of scattered junk
    connect_voxel = max(diag * 0.02, 1e-6)   # ~2% of bbox diagonal
    # Final voxel for low-poly resolution (main knob)
    final_voxel   = max(diag * 0.008, 1e-6)  # ~0.8% of bbox diagonal

    # Radii for denoising / normals
    r_large = final_voxel * 5.0

    # If your object has truly disconnected parts (e.g., 2 wheels not touching),
    # increase keep_top_k to 2 or 3.
    keep_top_k_components = 1

    # Target triangle budget for decimation
    target_tris = 8000  # try 2000-20000

    print(f"[INFO] bbox_diag={diag:.6g}")
    print(f"[PARAM] connect_voxel={connect_voxel:.6g}  final_voxel={final_voxel:.6g}  target_tris={target_tris}")

    # ---- QUICK FILTER FIRST (prevents big blob) ----
    # Light SOR first helps voxel connectivity not get linked by extreme outliers.
    print("[DENOISE] Light SOR (pre-filter)")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    print("[FILTER] Keep largest connected voxel component(s)")
    pcd = keep_largest_voxel_component(pcd, voxel_size=connect_voxel, keep_top_k=keep_top_k_components)

    # ---- Second pass denoise (now behaves better) ----
    print("[DENOISE] Stronger SOR (post core selection)")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.3)

    print("[DENOISE] ROR (post core selection)")
    pcd, _ = pcd.remove_radius_outlier(nb_points=12, radius=r_large)

    # ---- Downsample for low-poly control ----
    print("[DOWNSAMPLE] Voxel downsample (final resolution)")
    pcd = pcd.voxel_down_sample(voxel_size=final_voxel)

    cleaned_pcd_path = os.path.join(here, "cleaned_points.ply")
    o3d.io.write_point_cloud(cleaned_pcd_path, pcd)
    print(f"[SAVE] {cleaned_pcd_path}")

    # ---- Normals (for BPA/Poisson; alpha doesn't strictly need them but it's fine) ----
    print("[NORMALS] Estimate + orient normals")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=r_large, max_nn=60)
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(50)
    except Exception:
        pass

    # ---- Mesh ----
    print("[MESH] Build mesh (Alpha -> BPA -> Poisson fallback)")
    mesh, method = try_build_mesh(pcd, final_voxel=final_voxel)
    print(f"[MESH] Method: {method}")

    # Keep largest mesh component (kills leftover small blobs)
    mesh = keep_largest_mesh_component(mesh)
    mesh = clean_mesh(mesh)

    # ---- Decimate to target tris ----
    if len(mesh.triangles) > target_tris:
        print(f"[SIMPLIFY] Decimation: {len(mesh.triangles)} -> {target_tris}")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_tris))
        mesh = clean_mesh(mesh)

    mesh.compute_vertex_normals()

    out_ply = os.path.join(here, "mesh_lowpoly.ply")
    out_glb = os.path.join(here, "mesh_lowpoly.glb")

    o3d.io.write_triangle_mesh(out_ply, mesh)
    print(f"[SAVE] {out_ply}")

    try:
        o3d.io.write_triangle_mesh(out_glb, mesh, write_triangle_uvs=False)
        print(f"[SAVE] {out_glb}")
    except Exception as e:
        print(f"[WARN] GLB export failed (PLY is saved): {e}")

    print("[DONE]")


if __name__ == "__main__":
    main()
