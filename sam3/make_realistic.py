#!/usr/bin/env python3
"""
run_controlnet_on_sketches.py

Walks:

    sketches/{object_type}/individual_object/{x}/

For each view image (e.g. view0.png ... view5.png) it runs
SDXL+ControlNet via `generate_variants` using a single prompt
that makes the object as realistic as possible, and saves to:

    sketches/{object_type}/individual_object/{x}/realistic/
        view0_ctrl_0.png
        view1_ctrl_0.png
        ...
"""

from pathlib import Path

from controlnet_variants import generate_variants  # same file that defines generate_variants


# Root with object subfolders, matching your SAM3 script
SKETCHES_ROOT = Path("sketches")

# Single prompt: "make the object as realistic as possible"
REALISTIC_PROMPT = (
    "white background, realistic, studio lighting, "
    "a highly detailed, photorealistic rendering of the object, "
    "realistic materials and lighting, high resolution, ultra realistic"
)


def is_view_image(path: Path) -> bool:
    """Return True if this is a view image we want to process."""
    if not path.is_file():
        return False
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return False
    # avoid re-processing controlnet outputs if they end up in same dir for any reason
    if path.name.startswith("ctrl_") or "_ctrl_" in path.name:
        return False
    return True


def process_instance_dir(instance_dir: Path) -> None:
    """
    Process one instance folder:

        sketches/{object_type}/individual_object/{x}/

    For each view*.png (or any image file there), create:

        {instance_dir}/realistic/viewX_ctrl_0.png
    """
    realistic_dir = instance_dir / "realistic"
    realistic_dir.mkdir(exist_ok=True)

    # iterate files directly under instance_dir (view0.png, view1.png, ...)
    for img_path in sorted(instance_dir.iterdir()):
        if not is_view_image(img_path):
            continue

        view_stem = img_path.stem  # e.g. "view0"

        # one variant per view: prompt "make it realistic"
        saved_paths = generate_variants(
            input_path=str(img_path),
            out_dir=str(realistic_dir),
            style_prompts=[REALISTIC_PROMPT],
            seed=2025,
        )

        # Rename outputs so they don't overwrite each other:
        # ctrl_0.png -> view0_ctrl_0.png, etc.
        for idx, saved in enumerate(saved_paths):
            p = Path(saved)
            new_name = f"{view_stem}_ctrl_{idx}{p.suffix}"
            new_path = p.with_name(new_name)

            # If it already exists for some reason, overwrite
            if new_path.exists():
                new_path.unlink()

            p.rename(new_path)


def main() -> None:
    if not SKETCHES_ROOT.exists():
        raise SystemExit(f"{SKETCHES_ROOT} not found")

    # Walk object types: sketches/{object_type}/
    for obj_dir in sorted(SKETCHES_ROOT.iterdir()):
        if not obj_dir.is_dir():
            continue

        inst_root = obj_dir / "individual_object"
        if not inst_root.is_dir():
            continue  # skip object types without individual_object/

        # Walk instances: sketches/{object_type}/individual_object/{x}/
        for instance_dir in sorted(inst_root.iterdir()):
            if not instance_dir.is_dir():
                continue
            process_instance_dir(instance_dir)


if __name__ == "__main__":
    main()
