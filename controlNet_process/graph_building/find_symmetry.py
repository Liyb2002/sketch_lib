#!/usr/bin/env python3
# graph_building/find_symmetry.py

import re
from typing import Dict, List, Tuple, Any


_SUFFIX_RE = re.compile(r"^(.*)_([0-9]+)$")


def _base_name(name: str) -> str:
    """
    wheel_0 -> wheel
    unknown_0 -> unknown   (we keep it, but you may want to ignore unknown below)
    """
    m = _SUFFIX_RE.match(name)
    if not m:
        return name
    return m.group(1)


def find_symmetry(label_names: List[str], ignore_unknown: bool = True) -> Dict[str, Any]:
    """
    Args:
      label_names: list like ["wheel_0", "wheel_1", "frame_0", "unknown_0", ...]
      ignore_unknown: if True, do not create symmetry groups for "unknown_*"

    Returns:
      {
        "groups": { base_name: [label_names...] } only for groups with >= 2
        "pairs":  [ {"a": name_i, "b": name_j, "base": base}, ... ]  all pairs within each group
      }
    """
    groups: Dict[str, List[str]] = {}
    for n in label_names:
        base = _base_name(n)
        if ignore_unknown and base == "unknown":
            continue
        groups.setdefault(base, []).append(n)

    # keep only size>=2
    groups = {k: sorted(v) for k, v in groups.items() if len(v) >= 2}

    pairs = []
    for base, names in groups.items():
        names = sorted(names)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append({"a": names[i], "b": names[j], "base": base})

    return {"groups": groups, "pairs": pairs}
