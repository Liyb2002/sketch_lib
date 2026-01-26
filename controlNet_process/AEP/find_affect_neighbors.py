#!/usr/bin/env python3
# AEP/find_affect_neighbors.py
#
# Utility to find all neighbors affected by a target node name,
# based on symmetry pairs, attachments, and containment relations.

from __future__ import annotations

from typing import Any, Dict, List, Set


def find_affected_neighbors(constraints: Dict[str, Any], target: str) -> List[str]:
    """
    Find all neighbors directly connected to `target` by:
      - symmetry pairs
      - attachments edges
      - containment (outer/inner) relations

    Returns:
        Sorted list of unique neighbor node names (strings).
    """
    if not target:
        return []

    symmetry = constraints.get("symmetry", {}) or {}
    attachments = constraints.get("attachments", []) or []
    containment = constraints.get("containment", []) or []

    sym_neighbors: Set[str] = set()
    for p in symmetry.get("pairs", []) or []:
        a = p.get("a")
        b = p.get("b")
        if a == target and b:
            sym_neighbors.add(b)
        elif b == target and a:
            sym_neighbors.add(a)

    attach_neighbors: Set[str] = set()
    for e in attachments:
        a = e.get("a")
        b = e.get("b")
        if a == target and b:
            attach_neighbors.add(b)
        elif b == target and a:
            attach_neighbors.add(a)

    contain_neighbors: Set[str] = set()
    for c in containment:
        outer = c.get("outer")
        inner = c.get("inner")
        if outer == target and inner:
            contain_neighbors.add(inner)
        elif inner == target and outer:
            contain_neighbors.add(outer)

    all_neighbors = sorted(sym_neighbors | attach_neighbors | contain_neighbors)
    return all_neighbors
