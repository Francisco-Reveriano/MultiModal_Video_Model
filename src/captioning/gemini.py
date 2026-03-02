import glob
import json
import os
import time

import cv2
from google import genai


def caption_video_with_gemini(
    video_path: str,
    trigger_token: str,
    context: str,
    client: genai.Client,
) -> str | None:
    """Extract a mid-frame from a video and caption it with Gemini."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    _, buffer = cv2.imencode(".jpg", frame)
    img_bytes = buffer.tobytes()

    prompt = f"""You are captioning video frames for AI video generation model training.
CONTEXT: {context.strip()}
INSTRUCTIONS:
- Start with "{trigger_token}" as the subject identifier
- Describe: appearance, clothing, action/pose, setting, lighting, camera angle
- Describe implied motion
- 2-3 sentences, specific and visual
- Do NOT use real names, use "{trigger_token}" for the subject
- No opinions or subjective judgments"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {"inline_data": {"mime_type": "image/jpeg", "data": img_bytes}},
            prompt,
        ],
    )
    return response.text.strip()


def caption_all(
    video_dir: str,
    dataset_dir: str,
    trigger_token: str,
    context: str,
    api_key: str,
    per_video_context: dict | None = None,
    delay: float = 1.0,
) -> dict[str, str]:
    """Auto-caption all processed videos and write output files."""
    per_video_context = per_video_context or {}
    client = genai.Client(api_key=api_key)

    video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    print(f"Found {len(video_paths)} processed videos")

    if not video_paths:
        raise FileNotFoundError(f"No processed videos in {video_dir}")

    print("Auto-captioning...\n")
    captions = {}

    for i, video_path in enumerate(video_paths):
        filename = os.path.basename(video_path)
        print(f"  [{i + 1}/{len(video_paths)}] {filename}...", end=" ")

        ctx = per_video_context.get(filename, context)
        try:
            caption = caption_video_with_gemini(video_path, trigger_token, ctx, client)
            if caption:
                captions[filename] = caption
                print(f"OK\n     -> {caption}\n")
            else:
                captions[filename] = "[FAILED] Could not extract frame"
                print("FAIL: frame extraction failed\n")
        except Exception as e:
            captions[filename] = f"[FAILED] {e}"
            print(f"FAIL: {e}\n")

        time.sleep(delay)

    # Write outputs
    with open(os.path.join(dataset_dir, "captions.json"), "w") as f:
        json.dump(captions, f, indent=2)

    successful = {k: v for k, v in captions.items() if not v.startswith("[FAILED]")}

    with open(os.path.join(dataset_dir, "videos.txt"), "w") as f:
        for filename in successful:
            f.write(f"{video_dir}/{filename}\n")

    with open(os.path.join(dataset_dir, "prompts.txt"), "w") as f:
        for caption in successful.values():
            f.write(f"{caption}\n")

    print(f"{'=' * 60}")
    print(f"Captioned: {len(successful)}/{len(captions)} videos")

    return captions
