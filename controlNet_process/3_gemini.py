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


def build_prompts(object_description: str) -> list[str]:
    # 0) Strict photorealistic product shot (no shadows)
    p0 = (
        "Render the object in the input reference image with precise geometry and boundaries "
        "matching the sketch silhouette exactly. "
        f"Output a photorealistic product photo of a {object_description}, high quality, sharp focus, "
        "on a pure white background. "
        "ABSOLUTELY NO SHADOWS. NO DROP SHADOWS. No floor contact shadow. "
        "Do not add extra elements. Keep exact proportions. "
        "Main object color: green."
    )

    # 1) Premium catalog studio look (subtle lighting allowed but still clean)
    p1 = (
        "Using the input reference image, render the SAME object with extremely accurate geometry "
        "and clean edges. The object should look from the 20th century"
        f"Create a premium studio catalog photo of a {object_description}. "
        "White seamless background, minimal soft studio lighting, but NO visible cast shadow and NO drop shadow. "
        "High-end materials, realistic reflections but not exaggerated. "
        "No extra props, no text, no watermark. "
        "Main object color: red."
    )

    # 2) CAD / clay render style (very geometry-faithful)
    p2 = (
        "Render the object in the input reference image with exact geometry and proportions preserved. "
        f"Create a photorealistic cyberpunk-style product photograph of a {object_description}. "
        "Futuristic neon-accent lighting, subtle colored rim lights (cyan and magenta tones), but keep the background clean and near-white. "
        "NO strong shadows, NO drop shadows, no clutter. "
        "High realism, cinematic lighting, detailed materials, sci-fi industrial aesthetic. "
        "Main object color: blue, with subtle cyberpunk lighting reflections."
    )


    return [p0, p1, p2]


def _pil_from_part(part):
    inline = getattr(part, "inline_data", None)
    if inline is None:
        return None

    data = getattr(inline, "data", None)
    if data is None:
        return None

    if isinstance(data, str):
        import base64
        data = base64.b64decode(data)

    return PILImage.open(io.BytesIO(data)).convert("RGB")


def extract_first_pil_image(response):
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


def process_one_image(api_key: str, model: str, prompt: str, in_path: Path, out_path: Path):
    try:
        input_img = PILImage.open(in_path).convert("RGB")
        target_size = input_img.size  # (W,H)

        # per-thread client
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
    ap.add_argument("--out_root", default="sketch/views_realistic")
    ap.add_argument(
        "--model",
        default="gemini-3-pro-image-preview",
        choices=["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],
    )
    ap.add_argument("--object_description", default="product")
    ap.add_argument("--max_workers", type=int, default=6)
    ap.add_argument("--num_prompts", type=int, default=3, choices=[1, 2, 3])
    args = ap.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY (export it or put it in .env).")

    views_dir = Path(args.views_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not views_dir.exists():
        raise FileNotFoundError(f"Missing folder: {views_dir}")

    in_paths = sorted(views_dir.glob("*.png"))
    if len(in_paths) == 0:
        raise FileNotFoundError(f"No .png files found in {views_dir}")

    prompts = build_prompts(args.object_description)[: args.num_prompts]

    print(f"[INFO] Found {len(in_paths)} images in {views_dir}")
    print(f"[INFO] Output root: {out_root}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Workers: {args.max_workers}")
    print(f"[INFO] Prompt sets: {len(prompts)} -> folders 0..{len(prompts)-1}")

    # We keep each prompt-set processed sequentially (to reduce rate-limit spikes),
    # but each set does parallel per-view.
    for i, prompt in enumerate(prompts):
        out_dir = out_root / str(i)
        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"[PROMPT SET {i}] Writing to {out_dir}")
        print("=" * 80)

        futures = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            for in_path in in_paths:
                out_path = out_dir / in_path.name
                futures.append(ex.submit(process_one_image, api_key, args.model, prompt, in_path, out_path))

            ok_count = 0
            for fut in as_completed(futures):
                in_path, ok, msg = fut.result()
                status = "OK" if ok else "FAIL"
                print(f"[{status}] set={i} {in_path.name} — {msg}")
                ok_count += int(ok)

        print(f"[DONE] Prompt set {i}: {ok_count}/{len(in_paths)} succeeded.")

    print("\n[ALL DONE] Finished all prompt sets.")


if __name__ == "__main__":
    main()
