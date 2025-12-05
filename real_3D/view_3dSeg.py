#!/usr/bin/env python3
"""
view_overlap3d_strong.py

Aggressive viewer for the 3D overlap .ply files produced by verify_and_overlay.py.

- Forces use of stored colors (vertex or point).
- Prints diagnostics about color ranges.
- For meshes, converts to a point cloud with those colors so Open3D
  has no excuse to ignore them.

Usage:
    python view_overlap3d_strong.py path/to/file1.ply [path/to/file2.ply ...]
    python view_overlap3d_strong.py      # auto-discovers final_overlays/*overlap3d*.ply
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

DEFAULT_OUTPUT_DIR = Path("final_overlays")


def load_raw_geometry(path: Path):
    """Try mesh first, then point cloud."""
    print(f"\n==========")
    print(f"Loading: {path.resolve()}")

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.is_empty():
        print(f"   Detected: TriangleMesh")
        print(f"   Vertices:  {len(mesh.vertices)}  Triangles: {len(mesh.triangles)}")
        print(f"   Has vertex colors: {mesh.has_vertex_colors()}")
        return mesh

    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.is_empty():
        print(f"   Detected: PointCloud")
        print(f"   Points: {len(pcd.points)}")
        print(f"   Has colors: {pcd.has_colors()}")
        return pcd

    raise RuntimeError(f"Could not load a valid mesh or point cloud from {path}")


def ensure_pointcloud_with_colors(geom):
    """
    Convert whatever we have (mesh or pcd) into a PointCloud that definitely
    has color values attached.
    """
    if isinstance(geom, o3d.geometry.PointCloud):
        pts = np.asarray(geom.points)
        cols = np.asarray(geom.colors) if geom.has_colors() else None
    elif isinstance(geom, o3d.geometry.TriangleMesh):
        # Use vertices as points
        pts = np.asarray(geom.vertices)
        if geom.has_vertex_colors():
            cols = np.asarray(geom.vertex_colors)
        else:
            cols = None
    else:
        raise TypeError("Unknown geometry type.")

    if pts.shape[0] == 0:
        raise RuntimeError("Geometry has no points/vertices.")

    if cols is None or cols.shape[0] == 0:
        print("   ⚠️ No colors found in file — assigning gray and red randomly just to test.")
        cols = np.ones((pts.shape[0], 3), dtype=np.float32) * 0.7
    else:
        # Print stats so we know colors are actually there
        print(f"   Color stats: min={cols.min(axis=0)}, max={cols.max(axis=0)}")
        # If colors look like 0/255, normalize to [0,1]
        if cols.max() > 1.5:
            cols = cols / 255.0
            print("   Colors appear to be in 0-255; normalizing to 0-1.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    return pcd


def show_pointcloud(pcd, title: str):
    """
    Visualize a point cloud with explicit colors and a white background.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=800, height=800)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])  # white
    opt.point_size = 3.0

    # Center / zoom camera a bit
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = float(np.linalg.norm(extent)) if np.linalg.norm(extent) > 0 else 1.0

    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_up([0, 0, 1])
    ctr.set_front([0.5, 0.5, 0.5])
    ctr.set_zoom(0.7 if radius > 0 else 1.0)

    vis.run()
    vis.destroy_window()


def main():
    args = sys.argv[1:]

    if args:
        ply_paths = [Path(p) for p in args]
    else:
        if not DEFAULT_OUTPUT_DIR.exists():
            print(f"❌ No args and default folder not found: {DEFAULT_OUTPUT_DIR.resolve()}")
            return

        ply_paths = sorted(DEFAULT_OUTPUT_DIR.glob("*overlap3d*.ply"))
        if not ply_paths:
            print(f"❌ No '*overlap3d*.ply' files found in {DEFAULT_OUTPUT_DIR.resolve()}")
            return

    print(f"Found {len(ply_paths)} file(s) to view.")

    for path in ply_paths:
        if not path.exists():
            print(f"⚠️ Skipping missing file: {path}")
            continue

        try:
            geom = load_raw_geometry(path)
        except RuntimeError as e:
            print(f"⚠️ {e}")
            continue

        pcd = ensure_pointcloud_with_colors(geom)

        print(f"   PointCloud ready: {len(pcd.points)} points")
        print(f"   Has colors: {pcd.has_colors()}")

        # Show one at a time; close the window to move to the next
        show_pointcloud(pcd, title=f"Overlap 3D (point colors) - {path.name}")


if __name__ == "__main__":
    main()
