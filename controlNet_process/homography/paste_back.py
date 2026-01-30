#!/usr/bin/env python3
# homography/paste_back.py
#
# Read + print hierarchy_tree.json (adjacency dict format):
# {
#   "wheel_0": {"parent": null, "children": ["fender_0", ...]},
#   ...
# }
#
# Input:
#   sketch/AEP/hierarchy_tree.json
#
# Output:
#   prints root label, indented tree, and parent->child edges

import os
import json
from typing import Any, Dict, List, Optional, Tuple, Set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIERARCHY_PATH = os.path.join(ROOT, "sketch", "AEP", "hierarchy_tree.json")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _find_roots(tree: Dict[str, Any]) -> List[str]:
    roots = []
    for label, info in tree.items():
        parent = None
        if isinstance(info, dict):
            parent = info.get("parent", None)
        if parent is None:
            roots.append(label)
    return roots


def _children_of(tree: Dict[str, Any], label: str) -> List[str]:
    info = tree.get(label, {})
    if not isinstance(info, dict):
        return []
    ch = info.get("children", [])
    if ch is None:
        return []
    if isinstance(ch, list):
        return [str(x) for x in ch]
    return []


def extract_parent_child_edges(tree: Dict[str, Any]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for parent, info in tree.items():
        for child in _children_of(tree, parent):
            edges.append((parent, child))
    return edges


def _print_tree(tree: Dict[str, Any], root: str):
    visited: Set[str] = set()

    def dfs(node: str, indent: int):
        # avoid cycles / duplicates (just in case file is imperfect)
        if node in visited:
            print("  " * indent + f"- {node} (visited)")
            return
        visited.add(node)

        print("  " * indent + f"- {node}")
        for child in _children_of(tree, node):
            dfs(child, indent + 1)

    dfs(root, 0)


def main():
    print("\n" + "=" * 80)
    print("STEP 3: Reading hierarchy tree (root anchored, children movable)")
    print("=" * 80)

    if not os.path.exists(HIERARCHY_PATH):
        print(f"ERROR: hierarchy tree not found: {HIERARCHY_PATH}")
        return

    tree = _load_json(HIERARCHY_PATH)
    if not isinstance(tree, dict) or len(tree) == 0:
        print(f"ERROR: hierarchy_tree.json is not a non-empty dict: {HIERARCHY_PATH}")
        return

    roots = _find_roots(tree)
    edges = extract_parent_child_edges(tree)

    print(f"Hierarchy file: {HIERARCHY_PATH}")
    print(f"Num nodes: {len(tree)}")
    print(f"Num parent->child edges: {len(edges)}")

    if len(roots) == 0:
        print("ERROR: could not find any root (no node has parent=null).")
        return

    if len(roots) > 1:
        print(f"WARNING: multiple roots found: {roots}")
        print("Printing each root as its own tree.\n")

    for r in roots:
        print(f"\nRoot label: {r}")
        print("Tree:")
        _print_tree(tree, r)

    print("\nEdges:")
    for p, c in edges:
        print(f"  {p} -> {c}")

    print("=" * 80)


if __name__ == "__main__":
    main()
