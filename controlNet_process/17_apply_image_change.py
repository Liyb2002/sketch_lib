#!/usr/bin/env python3
# 17_apply_image_change.py
"""
Compute per-view updated 2D bboxes by applying tentative bbox edits onto real bboxes.

Fix requested:
- Labels in changed_labels_2d_bbox.json look like "wheel_0_0", but 2D segmentation uses "wheel_0".
- Therefore: when reading labels from changed_labels_2d_bbox.json, strip the trailing "_<digits>".
  (e.g., wheel_0_0 -> wheel_0, unknown_10_0 -> unknown_10)

Inputs:
- Tentative per-view changes (normalized to BEFORE-projected bbox):
    sketch/target_edit/2d_projection/view_{x}/changed_labels_2d_bbox.json

- Real bboxes per view (SAM):
    sketch/segmentation/view_{x}/all_components_bbox.json
  (also supports legacy all_components_bbx.json)

Output (per view):
    sketch/final_results/view_{x}/bbox_edits.json

Output schema:
{
  "view": "view_0",
  "image": "...",
  "image_size": {...},
  "changed_labels": [
    {
      "label": "wheel_0",
      "original_box_xyxy": [xmin,ymin,xmax,ymax],
      "new_box_xyxy": [xmin,ymin,xmax,ymax],
      "tentative_change_normalized": {scale_x, scale_y, trans_x, trans_y},
      "source_files": {...}
    },
    ...
  ]
}

Transform application rule:
Given real bbox B with center c, size (w,h),
and tentative change params (scale_x, scale_y, trans_x, trans_y)
where trans_* are normalized to bbox size,
we compute:
  c' = c + (trans_x * w, trans_y * h)
  w' = w * scale_x
  h' = h * scale_y
  updated bbox = [c'.x - w'/2, c'.y - h'/2, c'.x + w'/2, c'.y + h'/2]
"""

import os
import json
import argparse
from typing import Any, Dict, List


# ------------------------ IO ------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------ Label normalization ------------------------

def strip_trailing_index(label: str) -> str:
    """
    Strip the trailing "_<digits>" chunk if present.

    Examples:
      wheel_0_0     -> wheel_0
      unknown_10_0  -> unknown_10
      seat_0        -> seat       (NOTE: ONLY if last token is digits and there are >=2 parts)
    But for 2D labels like "seat_0", you usually do NOT want to strip.
    So we apply this ONLY to labels coming from changed_labels_2d_bbox.json.
    """
    parts = label.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return label


# ------------------------ Apply transform ------------------------

def apply_tentative_to_real_bbox_xyxy(
    box_xyxy: List[float],
    change: Dict[str, float],
) -> List[float]:
    xmin, ymin, xmax, ymax = map(float, box_xyxy)
    w = xmax - xmin
    h = ymax - ymin
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    sx = float(change.get("scale_x", 1.0))
    sy = float(change.get("scale_y", 1.0))
    tx = float(change.get("trans_x", 0.0))
    ty = float(change.get("trans_y", 0.0))

    cx2 = cx + tx * w
    cy2 = cy + ty * h
    w2 = w * sx
    h2 = h * sy

    xmin2 = cx2 - 0.5 * w2
    ymin2 = cy2 - 0.5 * h2
    xmax2 = cx2 + 0.5 * w2
    ymax2 = cy2 + 0.5 * h2

    return [float(xmin2), float(ymin2), float(xmax2), float(ymax2)]


def find_segmentation_bbox_file(view_dir: str) -> str:
    cands = [
        os.path.join(view_dir, "all_components_bbox.json"),  # your real filename
        os.path.join(view_dir, "all_components_bbx.json"),   # legacy
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return cands[0]


# ------------------------ Main ------------------------

def main(
    num_views: int,
    tentative_dir: str,
    segmentation_dir: str,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    for vid in range(num_views):
        view_name = f"view_{vid}"

        tentative_path = os.path.join(tentative_dir, view_name, "changed_labels_2d_bbox.json")
        view_seg_dir = os.path.join(segmentation_dir, view_name)
        seg_path = find_segmentation_bbox_file(view_seg_dir)

        if not os.path.isfile(tentative_path):
            raise FileNotFoundError(f"Missing tentative file: {tentative_path}")
        if not os.path.isfile(seg_path):
            raise FileNotFoundError(f"Missing segmentation bbox file: {seg_path}")

        tentative = load_json(tentative_path)
        seg = load_json(seg_path)

        # Build label -> tentative change map (valid only)
        # IMPORTANT: strip trailing index for labels coming from tentative file
        t_labels = tentative.get("labels", {})
        label2change: Dict[str, Dict[str, float]] = {}

        if isinstance(t_labels, dict):
            for raw_label, info in t_labels.items():
                if not isinstance(info, dict):
                    continue
                if not info.get("valid", False):
                    continue
                ch = info.get("change_from_before_to_after_normalized", None)
                if not isinstance(ch, dict):
                    continue

                # FIX: wheel_0_0 -> wheel_0
                norm_label = strip_trailing_index(str(raw_label))

                label2change[norm_label] = {
                    "scale_x": float(ch.get("scale_x", 1.0)),
                    "scale_y": float(ch.get("scale_y", 1.0)),
                    "trans_x": float(ch.get("trans_x", 0.0)),
                    "trans_y": float(ch.get("trans_y", 0.0)),
                }

        detections = seg.get("detections", [])
        if not isinstance(detections, list):
            raise ValueError(f"Invalid segmentation schema in {seg_path}: 'detections' must be a list")

        changed_labels_out = []
        for det in detections:
            if not isinstance(det, dict):
                continue

            # DO NOT strip for detection labels (they are already like wheel_0)
            label = str(det.get("label", ""))

            change = label2change.get(label, None)
            if change is None:
                continue  # not changed in 3D (or invalid projection)

            box_xyxy = det.get("box_xyxy", None)
            if not (isinstance(box_xyxy, list) and len(box_xyxy) == 4):
                continue

            changed_labels_out.append({
                "label": label,
                "original_box_xyxy": [float(x) for x in box_xyxy],
                "new_box_xyxy": apply_tentative_to_real_bbox_xyxy(box_xyxy, change),
                "tentative_change_normalized": change,
                "source_files": {
                    "segmentation_bbox_json": seg_path,
                    "tentative_change_json": tentative_path,
                }
            })

        out_obj = {
            "view": seg.get("view", view_name),
            "image": seg.get("image", None),
            "image_size": seg.get("image_size", None),
            "changed_labels": changed_labels_out,
        }

        view_out_dir = os.path.join(out_dir, view_name)
        os.makedirs(view_out_dir, exist_ok=True)
        out_path = os.path.join(view_out_dir, "bbox_edits.json")
        save_json(out_path, out_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_views", type=int, default=6)

    parser.add_argument(
        "--tentative_dir",
        default="sketch/target_edit/2d_projection",
        help="Folder containing view_{x}/changed_labels_2d_bbox.json",
    )
    parser.add_argument(
        "--segmentation_dir",
        default="sketch/segmentation",
        help="Folder containing view_{x}/all_components_bbox.json",
    )
    parser.add_argument(
        "--out_dir",
        default="sketch/final_results",
        help="Output folder: will create view_{x}/bbox_edits.json",
    )

    args = parser.parse_args()

    main(
        num_views=args.num_views,
        tentative_dir=args.tentative_dir,
        segmentation_dir=args.segmentation_dir,
        out_dir=args.out_dir,
    )
