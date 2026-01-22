import json

def load_aep_changes(path):
    with open(path, "r") as f:
        return json.load(f)

def list_obb_changes(aep):
    """
    Returns list of dicts:
      - target change has before_obb + after_obb
      - neighbor changes have after_obb only (before_obb=None)
    """
    out = []

    # 1) target edit (has before + after)
    tgt = aep["target"]
    ch = aep["target_edit"]["change"]
    out.append({
        "name": tgt,
        "role": "target",
        "before_obb": ch["before_obb"],
        "after_obb": ch["after_obb"],
    })

    # 2) neighbor changes (after only)
    neigh = aep.get("neighbor_changes", {}) or {}
    for name, rec in neigh.items():
        out.append({
            "name": name,
            "role": "neighbor",
            "before_obb": None,             # not in this file
            "after_obb": rec["after_obb"],
            "kind": rec.get("kind"),
            "case": rec.get("case"),
        })

    return out

# quick test
if __name__ == "__main__":
    aep = load_aep_changes("sketch/AEP/aep_changes.json")
    changes = list_obb_changes(aep)
    print("num changes:", len(changes))
    print([c["name"] for c in changes])
