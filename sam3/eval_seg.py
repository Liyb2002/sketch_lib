#!/usr/bin/env python3
"""
Count how many segments each object has, and totals per object type.

Folder structure:
sketches/{object_type}/individual_object/{object_id}/view_*/ *.png
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKETCHES = ROOT / "sketches"

dataset_total_segments = 0

print("\n========================================")
print("Segment Count Per Object")
print("========================================\n")

for object_type_folder in SKETCHES.iterdir():
    if not object_type_folder.is_dir():
        continue

    individual_root = object_type_folder / "individual_object"
    if not individual_root.exists():
        continue

    print(f"=== Object type: {object_type_folder.name} ===")

    object_type_total = 0  # total segments for this object type

    for object_folder in sorted(individual_root.iterdir(), key=lambda p: p.name):
        if not object_folder.is_dir():
            continue

        obj_name = object_folder.name
        seg_count = 0

        # Count pngs inside each view_* folder
        for view_folder in object_folder.iterdir():
            if view_folder.is_dir() and view_folder.name.startswith("view_"):
                seg_count += len(list(view_folder.glob("*.png")))

        object_type_total += seg_count
        print(f"{obj_name}: {seg_count} segments")

    dataset_total_segments += object_type_total
    print(f"Total segments in {object_type_folder.name}: {object_type_total}\n")

print("========================================")
print(f"TOTAL SEGMENTS IN DATASET: {dataset_total_segments}")
print("========================================")
