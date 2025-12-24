import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def _aabb_gap_distance(a_min, a_max, b_min, b_max):
    dx = max(0.0, max(b_min[0] - a_max[0], a_min[0] - b_max[0]))
    dy = max(0.0, max(b_min[1] - a_max[1], a_min[1] - b_max[1]))
    dz = max(0.0, max(b_min[2] - a_max[2], a_min[2] - b_max[2]))
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def _aabb_intersection(a_min, a_max, b_min, b_max):
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    return inter_min, inter_max


def _count_points_in_box(points_xyz, mn, mx):
    mask = np.all((points_xyz >= mn) & (points_xyz <= mx), axis=1)
    return int(np.count_nonzero(mask))


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def _vis_points_with_boxes(points, boxes):
    """
    boxes: list of (min_xyz, max_xyz)
    """
    geoms = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    geoms.append(pcd)

    for mn, mx in boxes:
        corners = np.array([
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]],
        ])

        lines = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ]

        box = o3d.geometry.LineSet()
        box.points = o3d.utility.Vector3dVector(corners)
        box.lines  = o3d.utility.Vector2iVector(lines)
        box.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
        geoms.append(box)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="merge_unknowns debug: connecting surfaces"
    )


# ---------------------------------------------------------------------
# MAIN API (DO NOT RENAME)
# ---------------------------------------------------------------------
def merge_unknowns(
    unknown_entities,
    points,
    threshold_points=100,
    extend_frac=0.005,
    debug=False,
    vis=False,
):
    """
    Merge unknown clusters if:
      1) Their AABBs touch or overlap
      2) Their AABB intersection surface is extended by
         extend = extend_frac * avg(global_bbox_dim)
      3) Each side contributes >= threshold_points inside that volume
    """

    n = len(unknown_entities)
    if n == 0:
        return []

    pts = np.asarray(points)

    # global scale
    shape_min = pts.min(axis=0)
    shape_max = pts.max(axis=0)
    dims = shape_max - shape_min
    avg_dim = float((dims[0] + dims[1] + dims[2]) / 3.0)
    extend = extend_frac * avg_dim

    if debug:
        print(
            f"[merge_unknowns] avg_dim={avg_dim:.6f}, "
            f"extend={extend:.6f}, threshold={threshold_points}"
        )

    parent = np.arange(n, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    debug_boxes = []

    for i in range(n):
        ai_min = unknown_entities[i]["aabb_min"]
        ai_max = unknown_entities[i]["aabb_max"]
        pts_i = pts[unknown_entities[i]["idxs"]]

        for j in range(i + 1, n):
            aj_min = unknown_entities[j]["aabb_min"]
            aj_max = unknown_entities[j]["aabb_max"]

            # 1) must touch
            if _aabb_gap_distance(ai_min, ai_max, aj_min, aj_max) != 0.0:
                continue

            # 2) intersection surface
            inter_min, inter_max = _aabb_intersection(ai_min, ai_max, aj_min, aj_max)

            # find thin axis (contact normal)
            thickness = inter_max - inter_min
            axis = np.argmin(thickness)

            # 3) extend along normal
            ext_min = inter_min.copy()
            ext_max = inter_max.copy()
            ext_min[axis] -= extend
            ext_max[axis] += extend

            debug_boxes.append((ext_min.copy(), ext_max.copy()))

            # 4) count independently
            ci = _count_points_in_box(pts_i, ext_min, ext_max)
            cj = _count_points_in_box(
                pts[unknown_entities[j]["idxs"]], ext_min, ext_max
            )

            if debug:
                print(
                    f"[merge_unknowns] pair ({i},{j}) "
                    f"axis={axis} ci={ci} cj={cj}"
                )

            if ci >= threshold_points and cj >= threshold_points:
                union(i, j)

    if vis and len(debug_boxes) > 0:
        _vis_points_with_boxes(pts, debug_boxes)

    groups = {}
    for k in range(n):
        r = find(k)
        groups.setdefault(r, []).append(k)

    merged_groups = sorted(
        (sorted(v) for v in groups.values()),
        key=lambda g: g[0]
    )

    if debug:
        print(f"[merge_unknowns] merged_groups={merged_groups}")

    return merged_groups
