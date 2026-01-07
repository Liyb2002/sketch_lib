#!/usr/bin/env python3
import os
import io
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from PIL import Image as PILImage

from google import genai
from google.genai import types


def build_prompt(object_description: str) -> str:
    return (
        "Render the object in the input reference image with **precise geometry and boundaries** "
        f"matching the sketch. Output a **photorealistic product photo** of a {object_description}, "
        "high-quality, on a clean white background. "
        "**ABSOLUTELY NO SHADOWS. NO DROP SHADOWS. Pure white background.** "
        "The main color of the object should be in red."
    )


def _pil_from_part(part) -> PILImage.Image | None:
    inline = getattr(part, "inline_data", None)
    if inline is None:
        return None

    data = getattr(inline, "data", None)
    if data is None:
        return None

    # bytes or base64 string
    if isinstance(data, str):
        import base64
        data = base64.b64decode(data)

    return PILImage.open(io.BytesIO(data)).convert("RGB")


def extract_first_pil_image(response) -> PILImage.Image | None:
    parts = getattr(response, "parts", None)

    if not parts:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None)

    if not parts:
        return None

    for part in parts:
        pil = _pil_from_part(part)
        if pil is not None:
            return pil

    return None


def process_one_image(
    api_key: str,
    model: str,
    prompt: str,
    in_path: Path,
    out_path: Path,
) -> tuple[Path, bool, str]:
    """
    Returns (in_path, ok, message)
    """
    try:
        input_img = PILImage.open(in_path).convert("RGB")
        target_size = input_img.size  # (W,H)

        # Create a fresh client per task (thread-safe and simple)
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model,
            contents=[prompt, input_img],
            config=types.GenerateContentConfig(
                response_modalities=["Image"],
            ),
        )

        gen_img = extract_first_pil_image(response)
        if gen_img is None:
            return (in_path, False, "No image returned (text-only or empty response).")

        # Force exact same size as this view's input
        if gen_img.size != target_size:
            gen_img = gen_img.resize(target_size, resample=PILImage.Resampling.LANCZOS)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        gen_img.save(out_path)

        return (in_path, True, f"Saved {out_path} (size={gen_img.size})")

    except Exception as e:
        return (in_path, False, f"{type(e).__name__}: {e}")


def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--views_dir", default="sketch/views")
    ap.add_argument("--out_dir", default="sketch/views_realistic")
    ap.add_argument(
        "--model",
        default="gemini-3-pro-image-preview",
        choices=["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],
    )
    ap.add_argument("--object_description", default="product")
    ap.add_argument("--max_workers", type=int, default=6)  # you said there are 6 images
    args = ap.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY (export it or put it in .env).")

    views_dir = Path(args.views_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not views_dir.exists():
        raise FileNotFoundError(f"Missing folder: {views_dir}")

    # You said exactly 6 .png files
    in_paths = sorted(views_dir.glob("*.png"))
    if len(in_paths) == 0:
        raise FileNotFoundError(f"No .png files found in {views_dir}")

    prompt = build_prompt(args.object_description)

    print(f"[INFO] Found {len(in_paths)} images in {views_dir}")
    print(f"[INFO] Writing to {out_dir}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Parallel workers: {args.max_workers}")

    # Parallel processing (network-bound)
    futures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for in_path in in_paths:
            out_path = out_dir / in_path.name
            futures.append(
                ex.submit(
                    process_one_image,
                    api_key,
                    args.model,
                    prompt,
                    in_path,
                    out_path,
                )
            )

        ok_count = 0
        for fut in as_completed(futures):
            in_path, ok, msg = fut.result()
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {in_path.name} — {msg}")
            ok_count += int(ok)

    print(f"[DONE] {ok_count}/{len(in_paths)} succeeded.")


if __name__ == "__main__":
    main()
