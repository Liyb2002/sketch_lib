#!/usr/bin/env python3
"""
unify_cluster_labels.py

Given a regrouped cluster mapping JSON produced by cluster_grouping.py, e.g.:

  regrouped_clusters_ply/0_trellis_gaussian_cluster_label_mapping.json

this script builds a "unified" label mapping that:

  - For base_names that appear with ONLY one instance (e.g., only "backrest_0"
    and no "backrest_1" anywhere across views), it simply unions all clusters
    for that instance across views.

  - For base_names that appear with MULTIPLE instances (e.g., "leg_0",
    "leg_1", "leg_2", ...), it matches instance labels across views by
    maximizing cluster overlap:
        * Each unified object (e.g. a leg) can have at most one label per view
        * Total number of unified objects for that base_name is at most the
          maximum number of that base_name seen in any single view
          (e.g., if a view sees 4 legs, we create at most 4 unified legs).

The output is a JSON of the form:

{
  "object_name": "...",
  "num_clusters": <int>,
  "labels": [
    {
      "base_name": "backrest",
      "unified_name": "backrest_0",
      "views": { "0": "backrest_0", "2": "backrest_0" },
      "clusters": [ ... union of clusters ... ]
    },
    {
      "base_name": "leg",
      "unified_name": "leg_0_u",
      "views": { "0": "leg_0", "1": "leg_2", "3": "leg_1" },
      "clusters": [ ... union of clusters ... ]
    },
    ...
  ]
}
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Set


# ---------------------------------------------------------------------
# HARD-CODED PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

INPUT_PATH = ROOT / "regrouped_clusters_ply" / "0_trellis_gaussian_cluster_label_mapping.json"
OUTPUT_PATH = ROOT / "regrouped_clusters_ply" / "0_trellis_gaussian_cluster_label_mapping_unified.json"


def load_mapping(mapping_path: Path) -> Dict[str, Any]:
    with open(mapping_path, "r") as f:
        return json.load(f)


def collect_entries(mapping: Dict[str, Any]):
    """
    From the cluster_label_mapping.json structure:

    {
      "object_name": ...,
      "num_clusters": ...,
      "views": {
        "0": {
          "labels": [
            {
              "label_id": ...,
              "name": "leg_0",
              "base_name": "leg",
              "clusters": [...],
              ...
            },
            ...
          ]
        },
        "1": { ... }
      }
    }

    Build:
      base_to_entries: base_name -> list of entries
        where each entry is:
          {
            "base_name": str,
            "view": str,
            "instance": str,   # e.g. "leg_0"
            "clusters": set(int),
          }

      base_to_instance_names: base_name -> set of distinct instance names
    """
    views = mapping.get("views", {})
    base_to_entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    base_to_instance_names: Dict[str, Set[str]] = defaultdict(set)

    for view_id, vdata in views.items():
        labels = vdata.get("labels", [])
        for lab in labels:
            base_name = lab.get("base_name")
            instance = lab.get("name")
            if not base_name or not instance:
                continue
            clusters = set(lab.get("clusters", []))
            entry = {
                "base_name": base_name,
                "view": str(view_id),
                "instance": instance,
                "clusters": clusters,
            }
            base_to_entries[base_name].append(entry)
            base_to_instance_names[base_name].add(instance)

    return base_to_entries, base_to_instance_names


def unify_single_instance(base_name: str,
                          entries: List[Dict[str, Any]],
                          instance_name: str) -> Dict[str, Any]:
    """
    For base_names with only one instance across all views, e.g. "backrest_0",
    union all clusters across all views.
    """
    all_clusters: Set[int] = set()
    views_map: Dict[str, str] = {}

    for e in entries:
        all_clusters |= e["clusters"]
        # keep which instance label appeared in which view (for completeness)
        views_map[e["view"]] = e["instance"]

    unified = {
        "base_name": base_name,
        "unified_name": instance_name,  # keep the same instance name
        "views": views_map,
        "clusters": sorted(all_clusters),
    }
    return unified


def jaccard_similarity(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def unify_multi_instance(base_name: str,
                         entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For base_names with multiple instances (e.g., "leg_0", "leg_1"), unify
    entries across views by shared clusters.

    Strategy:
      - Let K = max number of that base_name in any single view.
      - Sort entries by descending cluster count (bigger supports first).
      - Maintain up to K "global instances" for this base_name.
      - For each entry:
          * Try to attach it to an existing global instance that does NOT yet
            contain this view, maximizing Jaccard overlap.
          * If best similarity > 0, attach there.
          * Else, if we still have < K global instances, start a new instance.
          * Else, skip this entry (ambiguous / extra, we ignore it).

      - Each global instance yields:
          {
            "base_name": base_name,
            "unified_name": f"{base_name}_{idx}_u",
            "views": { view_id: original_instance_name, ... },
            "clusters": sorted(union_of_clusters)
          }
    """
    # Determine K: max count per view
    per_view_counts: Dict[str, int] = defaultdict(int)
    for e in entries:
        per_view_counts[e["view"]] += 1
    K = max(per_view_counts.values()) if per_view_counts else 0
    if K == 0:
        return []

    # Sort entries by descending cluster count (larger supports first)
    entries_sorted = sorted(entries, key=lambda e: len(e["clusters"]), reverse=True)

    # Each global instance: dict with keys: id, views, clusters, members
    global_instances: List[Dict[str, Any]] = []

    for e in entries_sorted:
        v = e["view"]
        cset = e["clusters"]

        # Find best existing global instance that doesn't use this view yet
        best_g = None
        best_sim = 0.0
        for g in global_instances:
            if v in g["views"]:
                continue
            sim = jaccard_similarity(cset, g["clusters"])
            if sim > best_sim:
                best_sim = sim
                best_g = g

        assigned = False

        # If we have some non-zero overlap with an existing instance, attach
        if best_g is not None and best_sim > 0.0:
            best_g["views"].add(v)
            best_g["clusters"] |= cset
            best_g["members"].append(e)
            assigned = True
        else:
            # Otherwise, if we still have capacity (< K), create a new instance
            if len(global_instances) < K:
                g_new = {
                    "id": len(global_instances),
                    "views": {v},
                    "clusters": set(cset),
                    "members": [e],
                }
                global_instances.append(g_new)
                assigned = True

        # If not assigned (no overlap and already at K), we IGNORE this entry:
        # it's an ambiguous extra, we don't create new unified label for it.
        if not assigned:
            continue

    # Build unified label entries from global_instances
    unified_entries: List[Dict[str, Any]] = []
    for g in global_instances:
        if not g["members"]:
            continue

        # Map view -> whichever original instance name was used there
        views_map: Dict[str, str] = {}
        for m in g["members"]:
            views_map[m["view"]] = m["instance"]

        unified_name = f"{base_name}_{g['id']}_u"

        unified = {
            "base_name": base_name,
            "unified_name": unified_name,
            "views": views_map,
            "clusters": sorted(g["clusters"]),
        }
        unified_entries.append(unified)

    return unified_entries


def unify_all_labels(mapping: Dict[str, Any]) -> Dict[str, Any]:
    base_to_entries, base_to_instance_names = collect_entries(mapping)

    unified_labels: List[Dict[str, Any]] = []

    for base_name, entries in base_to_entries.items():
        instance_names = base_to_instance_names.get(base_name, set())
        if not entries:
            continue

        if len(instance_names) == 1:
            # Single instance across all views (e.g. "backrest_0")
            inst_name = next(iter(instance_names))
            unified = unify_single_instance(base_name, entries, inst_name)
            unified_labels.append(unified)
        else:
            # Multi-instance label (e.g., "leg_0", "leg_1", ...)
            multi_unified = unify_multi_instance(base_name, entries)
            unified_labels.extend(multi_unified)

    # Sort for nicer output: by base_name then unified_name
    unified_labels.sort(key=lambda x: (x["base_name"], x["unified_name"]))

    result = {
        "object_name": mapping.get("object_name"),
        "num_clusters": mapping.get("num_clusters"),
        "labels": unified_labels,
    }
    return result


def main():
    if not INPUT_PATH.is_file():
        print(f"❌ Input JSON not found: {INPUT_PATH}")
        return

    print(f"[INFO] Loading mapping from {INPUT_PATH}")
    mapping = load_mapping(INPUT_PATH)
    unified = unify_all_labels(mapping)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(unified, f, indent=2)

    print(f"[DONE] Wrote unified label→cluster mapping to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
