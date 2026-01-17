#!/usr/bin/env python3
# AEP/attachment.py
#
# For now: just print which attachment relations involve the target.

from typing import Dict, Any, List


def handle_attachments(
    target: str,
    att_edges: List[Dict[str, Any]],
) -> None:
    if len(att_edges) == 0:
        print(f"[AEP][ATTACH] target={target}: no attachment relations.")
        return

    print(f"[AEP][ATTACH] target={target}: {len(att_edges)} attachment relation(s) happen:")
    for e in att_edges:
        a = e.get("a", "?")
        b = e.get("b", "?")

        # face fields may or may not exist depending on which file produced them
        a_face = e.get("a_face", None)
        b_face = e.get("b_face", None)
        axis = e.get("axis", None)
        dist = e.get("distance", None)

        if a_face is not None and b_face is not None:
            print(f"  - {a}({a_face}) <-> {b}({b_face})   axis=u{axis} dist={dist}")
        else:
            print(f"  - {a} <-> {b}   dist={dist}")

    print("")  # spacing
