# AEP/attachment_scaling.py

from typing import Dict, Any, Tuple
import numpy as np


def scale_neighbor_obb(
    other_obb: Dict[str, Any],
    edit_face_normal: np.ndarray,
    scale_ratio: float,
    min_extent: float,
    edited_face_str: str,
    neighbor_name: str,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Stub scaling function.

    For now:
      - does NOT modify the OBB
      - only prints debug info
      - returns the original OBB unchanged

    Returns:
      after_obb, debug_info
    """

    if verbose:
        print("[AEP][attachment_scaling] scaling called!")
        print(f"[AEP][attachment_scaling] neighbor = {neighbor_name}")
        print(f"[AEP][attachment_scaling] edited_face = {edited_face_str}")
        print(f"[AEP][attachment_scaling] edit_face_normal = {edit_face_normal.tolist()}")
        print(f"[AEP][attachment_scaling] scale_ratio = {scale_ratio}")
        print(f"[AEP][attachment_scaling] min_extent = {min_extent}")

    debug_info = {
        "status": "stub_called",
        "edited_face": edited_face_str,
        "scale_ratio": float(scale_ratio),
    }

    # return unchanged
    return other_obb, debug_info
