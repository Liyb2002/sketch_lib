import numpy as np

def aabb_of_points(pts):
    """Calculate the axis-aligned bounding box (AABB) of a set of points."""
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return mn, mx

def aabb_gap_distance(a_min, a_max, b_min, b_max):
    """
    Euclidean distance between two AABBs (0 if they overlap/touch).
    """
    dx = max(0.0, max(b_min[0] - a_max[0], a_min[0] - b_max[0]))
    dy = max(0.0, max(b_min[1] - a_max[1], a_min[1] - b_max[1]))
    dz = max(0.0, max(b_min[2] - a_max[2], a_min[2] - b_max[2]))
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))

def get_points_in_box(center, size, points):
    """
    Count how many points are inside a box centered at `center` with side length `size`.
    """
    half_size = size / 2
    min_bound = center - half_size
    max_bound = center + half_size
    
    # Debugging: Print the box bounds
    print(f"[DEBUG] Center: {center}, Box Size: {size}")
    print(f"[DEBUG] Min Bound: {min_bound}, Max Bound: {max_bound}")

    inside_points = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    
    # Debugging: Check how many points are inside the box
    points_inside = np.sum(inside_points)
    print(f"[DEBUG] Points inside the box: {points_inside}")
    
    return points_inside

def merge_unknowns(unknown_entities, points, merge_gap=0.1, threshold=300, box_size=1):
    """
    Merge unknown clusters based on their bounding boxes and the number of points within the center box.
    Two clusters are merged if their bounding boxes are connected and the center box contains enough points.
    """
    merged_groups = []
    visited = [False] * len(unknown_entities)

    # Iterate over all unknown entities and check for merges
    for i, entity_i in enumerate(unknown_entities):
        if visited[i]:
            continue

        visited[i] = True
        merged = [i]  # Start with the current entity as a merged group

        for j, entity_j in enumerate(unknown_entities):
            if i == j or visited[j]:
                continue

            # Check if their bounding boxes are connected
            gap = aabb_gap_distance(entity_i["aabb_min"], entity_i["aabb_max"], entity_j["aabb_min"], entity_j["aabb_max"])
            if gap <= merge_gap:
                # Check if the center of the bounding boxes contains enough points
                center_i = (entity_i["aabb_min"] + entity_i["aabb_max"]) / 2
                center_j = (entity_j["aabb_min"] + entity_j["aabb_max"]) / 2
                center = (center_i + center_j) / 2

                points_in_box_i = get_points_in_box(center, box_size, points[entity_i["idxs"]])
                points_in_box_j = get_points_in_box(center, box_size, points[entity_j["idxs"]])
                total_points = points_in_box_i + points_in_box_j

                # Debugging: Print the points inside the box for each cluster
                print(f"[DEBUG] Total points in merged box: {total_points}")

                if total_points >= threshold:
                    # If enough points are in the box, merge the clusters
                    merged.append(j)
                    visited[j] = True

        # Add the merged group to the list
        merged_groups.append(merged)

    return merged_groups
