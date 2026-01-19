#!/usr/bin/env python3
"""
Print (a, b) pairs for ONLY:
- containment relations
- attachment relations

Reads: sketch/AEP/initial_constraints.json  (relative to this script's location)
"""

import os
import json


def _fmt_pair(r: dict) -> str:
    a = r.get("a", "<missing a>")
    b = r.get("b", "<missing b>")
    return f"{a}  <->  {b}"


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "sketch", "AEP", "initial_constraints.json")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # -------------------------
    # Attachments
    # -------------------------
    attachments = data.get("attachments", [])
    print("\n=== ATTACHMENTS ===")
    print(f"count: {len(attachments)}")
    for i, r in enumerate(attachments):
        rel_type = r.get("relation_type", "unknown")
        dist = r.get("distance", None)
        suffix = []
        suffix.append(f"type={rel_type}")
        if dist is not None:
            suffix.append(f"distance={dist}")
        print(f"[{i:03d}] {_fmt_pair(r)}    ({', '.join(suffix)})")

    # -------------------------
    # Containment
    # -------------------------
    containment = data.get("containment", [])
    print("\n=== CONTAINMENT ===")
    print(f"count: {len(containment)}")
    for i, r in enumerate(containment):
        # some pipelines store keys like: container/containee, outer/inner, parent/child
        # but you said your schema uses "a" and "b", so we print that first.
        print(f"[{i:03d}] {_fmt_pair(r)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
