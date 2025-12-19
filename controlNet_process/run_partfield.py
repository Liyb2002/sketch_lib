#!/usr/bin/env python3
import numpy as np
from pathlib import Path

import trimesh
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


ROOT = Path(__file__).resolve().parent
IN_GLB = ROOT / "sketch" / "3d_reconstruction" / "fused_model.ply"

OUT_DIR = IN_GLB.parent
OUT_LABELS = OUT_DIR / "clustering_k20.npy"
OUT_POINTS_PLY = OUT_DIR / "clustering_k20_points.ply"
OUT_MESH_GLB = OUT_DIR / "clustering_k20.glb"
OUT_MESH_PLY = OUT_DIR / "clustering_k20.ply"

K = 20
N_SAMPLE = 200_000   # for meshes: number of sampled surface points
N_POINTS_MAX = 500_000  # for point clouds: cap points to avoid huge RAM/time


def load_geometry_any(path: Path):
    """
    Returns either:
      - trimesh.Trimesh (with faces)
      - trimesh.points.PointCloud (vertices only)
    """
    loaded = trimesh.load(path, force=None)

    # Direct mesh
    if isinstance(loaded, trimesh.Trimesh):
        # Could be a "mesh" with 0 faces if file is point cloud-ish
        if len(loaded.vertices) > 0 and len(loaded.faces) > 0:
            return loaded
        # If no faces, treat as point cloud
        return trimesh.points.PointCloud(loaded.vertices)

    # Direct point cloud
    if isinstance(loaded, trimesh.points.PointCloud):
        if len(loaded.vertices) == 0:
            raise RuntimeError(f"Point cloud is empty: {path}")
        return loaded

    # Scene -> merge meshes if present, else fall back to point clouds
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Could not load geometry from: {path}")

        meshes = []
        pclouds = []

        for g in loaded.geometry.values():
            if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0:
                if len(g.faces) > 0:
                    meshes.append(g)
                else:
                    pclouds.append(trimesh.points.PointCloud(g.vertices))
            elif isinstance(g, trimesh.points.PointCloud) and len(g.vertices) > 0:
                pclouds.append(g)

        if meshes:
            merged = trimesh.util.concatenate(meshes)
            merged.process(validate=False)
            return merged

        if pclouds:
            verts = np.vstack([np.asarray(pc.vertices) for pc in pclouds])
            return trimesh.points.PointCloud(verts)

    raise RuntimeError(f"Unsupported geometry type loaded from: {path}")


def color_map_from_labels(labels: np.ndarray):
    labels = labels.astype(np.int64)
    uniq = np.unique(labels[labels >= 0])
    uniq = np.sort(uniq)
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 1))
    lab2rgb = {int(lab): (np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8)
               for i, lab in enumerate(uniq)}
    return lab2rgb


def export_colored_points_ply(points_xyz: np.ndarray, labels: np.ndarray, out_ply: Path):
    labels = labels.reshape(-1).astype(np.int64)
    assert points_xyz.shape[0] == labels.shape[0]

    lab2rgb = color_map_from_labels(labels)
    cols = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for i, lab in enumerate(labels):
        if lab >= 0:
            cols[i] = lab2rgb[int(lab)]

    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {points_xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]) + "\n"

    with open(out_ply, "w") as f:
        f.write(header)
        for p, c in zip(points_xyz, cols):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    print(f"[SAVE] {out_ply}")


def export_mesh_by_face_vote(mesh: trimesh.Trimesh,
                             points_xyz: np.ndarray,
                             point_labels: np.ndarray,
                             out_glb: Path,
                             out_ply: Path):
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces, dtype=np.int64)
    point_labels = point_labels.reshape(-1).astype(np.int64)
    assert points_xyz.shape[0] == point_labels.shape[0]

    prox = trimesh.proximity.ProximityQuery(mesh)
    closest, dist, face_id = prox.on_surface(points_xyz)

    nF = F.shape[0]
    face_label = np.full((nF,), -1, dtype=np.int64)

    buckets = [[] for _ in range(nF)]
    for i in range(points_xyz.shape[0]):
        fi = int(face_id[i])
        lab = int(point_labels[i])
        if lab >= 0:
            buckets[fi].append(lab)

    for fi in range(nF):
        if buckets[fi]:
            labs = np.array(buckets[fi], dtype=np.int64)
            face_label[fi] = np.bincount(labs).argmax()

    vert_label = np.full((V.shape[0],), -1, dtype=np.int64)
    incident = [[] for _ in range(V.shape[0])]
    for fi, tri in enumerate(F):
        lab = int(face_label[fi])
        if lab < 0:
            continue
        a, b, c = map(int, tri)
        incident[a].append(lab)
        incident[b].append(lab)
        incident[c].append(lab)

    for vi in range(V.shape[0]):
        if incident[vi]:
            labs = np.array(incident[vi], dtype=np.int64)
            vert_label[vi] = np.bincount(labs).argmax()

    lab2rgb = color_map_from_labels(vert_label)
    colors_rgb = np.zeros((V.shape[0], 3), dtype=np.uint8)
    for i, lab in enumerate(vert_label):
        if lab >= 0:
            colors_rgb[i] = lab2rgb[int(lab)]

    alpha = np.full((colors_rgb.shape[0], 1), 255, dtype=np.uint8)
    colors_rgba = np.concatenate([colors_rgb, alpha], axis=1)

    out_mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    out_mesh.visual.vertex_colors = colors_rgba

    out_mesh.export(out_glb)
    print(f"[SAVE] {out_glb}")
    out_mesh.export(out_ply)
    print(f"[SAVE] {out_ply}")


def main():
    if not IN_GLB.is_file():
        raise RuntimeError(f"Missing input: {IN_GLB}")

    geom = load_geometry_any(IN_GLB)

    # ---- Case A: Mesh ----
    if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
        mesh = geom
        print("[INFO] Loaded MESH:", IN_GLB)
        print("[INFO] Vertices:", len(mesh.vertices), "Faces:", len(mesh.faces))

        P = int(N_SAMPLE)
        points_xyz, _ = trimesh.sample.sample_surface(mesh, P)
        points_xyz = points_xyz.astype(np.float32)
        print("[INFO] Sampled surface points:", points_xyz.shape[0])

        km = MiniBatchKMeans(
            n_clusters=K,
            batch_size=8192,
            n_init="auto",
            max_iter=200,
            random_state=0,
        )
        labels = km.fit_predict(points_xyz).astype(np.int64)
        print("[INFO] Clusters:", len(np.unique(labels)))

        np.save(OUT_LABELS, labels)
        print(f"[SAVE] {OUT_LABELS}")

        export_colored_points_ply(points_xyz, labels, OUT_POINTS_PLY)
        export_mesh_by_face_vote(mesh, points_xyz, labels, OUT_MESH_GLB, OUT_MESH_PLY)

        print("\n✅ Done (mesh path).")
        return

    # ---- Case B: Point Cloud ----
    if isinstance(geom, trimesh.points.PointCloud):
        points_xyz = np.asarray(geom.vertices, dtype=np.float32)
        print("[INFO] Loaded POINT CLOUD:", IN_GLB)
        print("[INFO] Points:", points_xyz.shape[0])

        # Optional downsample if huge
        if points_xyz.shape[0] > N_POINTS_MAX:
            idx = np.random.RandomState(0).choice(points_xyz.shape[0], N_POINTS_MAX, replace=False)
            points_xyz = points_xyz[idx]
            print("[INFO] Downsampled to:", points_xyz.shape[0])

        km = MiniBatchKMeans(
            n_clusters=K,
            batch_size=8192,
            n_init="auto",
            max_iter=200,
            random_state=0,
        )
        labels = km.fit_predict(points_xyz).astype(np.int64)
        print("[INFO] Clusters:", len(np.unique(labels)))

        np.save(OUT_LABELS, labels)
        print(f"[SAVE] {OUT_LABELS}")

        # For point clouds, we only export colored points (no mesh faces to color)
        export_colored_points_ply(points_xyz, labels, OUT_POINTS_PLY)

        print("[INFO] Skipped mesh export: input has no faces (point cloud).")
        print("\n✅ Done (point-cloud path).")
        return

    raise RuntimeError("Loaded geometry is neither a usable mesh nor a point cloud.")


if __name__ == "__main__":
    main()
