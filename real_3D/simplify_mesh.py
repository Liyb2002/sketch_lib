#!/usr/bin/env python3
import numpy as np
import trimesh
from pathlib import Path

MAX_FACES = 5000

def simplify_mesh(input_path):
    input_path = Path(input_path).expanduser()
    output_path = input_path.with_name(
        input_path.stem + "_simplified" + input_path.suffix
    )

    print(f"Loading mesh: {input_path}")
    mesh = trimesh.load(input_path, force='mesh', process=False)

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Loaded mesh has no faces. Check the input file.")

    n_faces = len(mesh.faces)
    print(f"Before simplify: {len(mesh.vertices)} vertices | {n_faces} faces")

    if n_faces > MAX_FACES:
        print(f"Randomly subsampling faces down to {MAX_FACES}...")

        # Choose a subset of face indices
        idx = np.random.choice(n_faces, size=MAX_FACES, replace=False)
        faces_sub = mesh.faces[idx]

        # Collect the unique vertices actually used by these faces
        unique_verts, inverse = np.unique(faces_sub.reshape(-1), return_inverse=True)
        verts_sub = mesh.vertices[unique_verts]

        # Remap face indices to the compact vertex array
        faces_sub_remapped = inverse.reshape(-1, 3)

        # Build new mesh
        mesh_simplified = trimesh.Trimesh(
            vertices=verts_sub,
            faces=faces_sub_remapped,
            process=False
        )

        print(f"After simplify: {len(mesh_simplified.vertices)} vertices | {len(mesh_simplified.faces)} faces")
    else:
        print("Mesh already below threshold. No simplification applied.")
        mesh_simplified = mesh

    mesh_simplified.export(output_path)
    print(f"Saved simplified mesh to: {output_path}")

if __name__ == "__main__":
    simplify_mesh("~/Desktop/sketch_lib/real_3D/trellis_outputs/0_trellis_raw.glb")
