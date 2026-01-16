#!/usr/bin/env python3
# graph_building/object_space.py

import os
import json
import numpy as np


def _ensure_right_handed(R: np.ndarray) -> np.ndarray:
    # R columns are axes. Ensure det(R) = +1
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R


def _fix_axis_signs_deterministic(R: np.ndarray) -> np.ndarray:
    """
    PCA eigenvectors have sign ambiguity. Make sign deterministic by anchoring
    each axis to the world basis: the largest-magnitude component of each axis
    is forced to be positive.
    """
    R2 = R.copy()
    for i in range(3):
        v = R2[:, i]
        j = int(np.argmax(np.abs(v)))
        if v[j] < 0:
            R2[:, i] *= -1.0
    R2 = _ensure_right_handed(R2)
    return R2


def compute_object_space(points_xyz: np.ndarray):
    """
    Return a dict:
      {
        "origin": [x,y,z],
        "axes": [[...],[...],[...]]   # 3x3, columns are object axes in world space
      }

    Object frame coordinates:
      p_local = (p_world - origin) @ axes
    since axes columns form an orthonormal basis.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("Need Nx3 points with N>=3 to compute object space.")

    origin = pts.mean(axis=0)
    X = pts - origin

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]  # columns are principal axes

    R = _fix_axis_signs_deterministic(R)

    return {
        "origin": origin.tolist(),
        "axes": R.tolist(),
    }


def world_to_object(points_xyz: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    return (pts - origin) @ axes


def object_to_world(points_local: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    pl = np.asarray(points_local, dtype=np.float64)
    return origin + pl @ axes.T


def save_object_space(path_json: str, obj_space: dict) -> None:
    os.makedirs(os.path.dirname(path_json), exist_ok=True)
    tmp = path_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj_space, f, indent=2)
    os.replace(tmp, path_json)


def load_object_space(path_json: str) -> dict:
    with open(path_json, "r") as f:
        return json.load(f)
