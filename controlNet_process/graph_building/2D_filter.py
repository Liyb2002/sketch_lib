#!/usr/bin/env python3
"""
graph_building/2D_filter.py

- Load sketch/AEP/initial_constraints.json
- Extract attachment pairs (a,b)
- Compute aggregated 2D relations across all views (mask neighbors)
- Find pairs that are in JSON attachments but not in 2D relations,
  BUT NEVER remove anything involving 'unknown_*'
- Remove those attachment entries (only non-unknown pairs)
- Save to: sketch/AEP/filtered_relations.json

Prints:
- counts
- list of removed unique pairs (non-unknown only)
- list of "missing but kept because unknown" pairs
"""

import os
import json
import importlib.util


def _load_module_from_path(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pair(a: str, b: str):
    return tuple(sorted((a, b)))


def _is_unknown(name: str) -> bool:
    return isinstance(name, str) and name.startswith("unknown_")


def filter_attachments_by_2d_relations(
    root_dir: str,
    seg_root: str,
    constraints_path: str,
    output_path: str,
    tol_ratio: float = 0.10,
    views=range(6),
):
    # Load aggregation module (filename starts with digit, so load by path)
    agg_path = os.path.join(root_dir, "graph_building", "2D_relations_aggregation.py")
    agg = _load_module_from_path(agg_path, "twoD_relations_aggregation")

    # 1) compute all-view 2D relations
    all_view_relations, rel_stats = agg.compute_all_view_relations(
        seg_root=seg_root,
        views=views,
        tol_ratio=tol_ratio,
    )

    # 2) read constraints json
    with open(constraints_path, "r") as f:
        data = json.load(f)

    attachments = data.get("attachments", [])
    attach_pairs = set()
    for rel in attachments:
        a = rel.get("a")
        b = rel.get("b")
        if a and b:
            attach_pairs.add(_pair(a, b))

    # 3) classify pairs "in json but not in 2D relations"
    missing_pairs_all = sorted([p for p in attach_pairs if p not in all_view_relations])

    missing_pairs_remove = []  # missing AND no unknowns
    missing_pairs_keep_unknown = []  # missing BUT involves unknown
    for a, b in missing_pairs_all:
        if _is_unknown(a) or _is_unknown(b):
            missing_pairs_keep_unknown.append((a, b))
        else:
            missing_pairs_remove.append((a, b))

    missing_remove_set = set(missing_pairs_remove)

    # 4) filter attachment entries (never remove unknown-related)
    filtered_attachments = []
    removed_entries = 0
    kept_unknown_missing_entries = 0

    for rel in attachments:
        a = rel.get("a")
        b = rel.get("b")
        if not a or not b:
            filtered_attachments.append(rel)
            continue

        # NEVER remove unknown-related
        if _is_unknown(a) or _is_unknown(b):
            # track if it was missing but kept
            if _pair(a, b) in set(missing_pairs_keep_unknown):
                kept_unknown_missing_entries += 1
            filtered_attachments.append(rel)
            continue

        # remove only if (a,b) missing in 2D and both non-unknown
        if _pair(a, b) in missing_remove_set:
            removed_entries += 1
            continue

        filtered_attachments.append(rel)

    # 5) write out filtered json
    out_data = dict(data)
    out_data["attachments"] = filtered_attachments

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=2)

    # 6) print summary (clean)
    print("=" * 100)
    print("2D FILTER SUMMARY (unknown_* attachments are NEVER removed)")
    print("-" * 100)
    print(f"constraints_in:  {constraints_path}")
    print(f"filtered_out:    {output_path}")
    print(f"seg_root:        {seg_root}")
    print(f"views:           {list(views)}")
    print(f"tol_ratio:       {tol_ratio}")
    print("-" * 100)
    print(f"2D relations (all views):       {rel_stats['relations_all_count']}")
    print(f"JSON attachment unique pairs:   {len(attach_pairs)}")
    print(f"JSON attachment raw entries:    {len(attachments)}")
    print("-" * 100)
    print(f"Missing pairs (JSON not in 2D): {len(missing_pairs_all)}")
    print(f"  removable (non-unknown):      {len(missing_pairs_remove)}")
    print(f"  kept (involves unknown_*):    {len(missing_pairs_keep_unknown)}")
    print("-" * 100)
    print(f"Attachment entries removed:     {removed_entries}")
    print(f"Missing-unknown entries kept:   {kept_unknown_missing_entries}")
    print(f"Attachment entries kept total:  {len(filtered_attachments)}")
    print("-" * 100)

    if missing_pairs_remove:
        print("Removed attachment pairs (unique, non-unknown):")
        for a, b in missing_pairs_remove:
            print(f"  {a}  <->  {b}")
    else:
        print("Removed attachment pairs (unique, non-unknown): (none)")

    if missing_pairs_keep_unknown:
        print("-" * 100)
        print("Missing in 2D but KEPT because unknown_* involved (unique):")
        for a, b in missing_pairs_keep_unknown:
            print(f"  {a}  <->  {b}")

    print("=" * 100)


def main():
    # This file lives in graph_building/, so project root is parent
    gb_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(gb_root)

    seg_root = os.path.join(project_root, "sketch", "segmentation_original_image")
    constraints_path = os.path.join(project_root, "sketch", "AEP", "initial_constraints.json")
    output_path = os.path.join(project_root, "sketch", "AEP", "filtered_relations.json")

    if not os.path.isdir(seg_root):
        raise FileNotFoundError(f"seg_root not found: {seg_root}")
    if not os.path.isfile(constraints_path):
        raise FileNotFoundError(f"constraints not found: {constraints_path}")

    filter_attachments_by_2d_relations(
        root_dir=project_root,
        seg_root=seg_root,
        constraints_path=constraints_path,
        output_path=output_path,
        tol_ratio=0.10,
        views=range(6),
    )


if __name__ == "__main__":
    main()
