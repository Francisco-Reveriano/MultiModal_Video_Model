"""
End-to-end dataset preparation pipeline.

Steps:
  1. Analyze raw videos
  2. Process (resize, crop, trim)
  3. Preview thumbnails
  4. Auto-caption with Gemini
  5. Validate final dataset

Usage:
  python -m scripts.prepare_dataset
  python -m scripts.prepare_dataset --skip-caption   # reuse existing captions
  python -m scripts.prepare_dataset --skip-preview    # headless / no display
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    CAPTION_CONTEXT,
    DATASET_DIR,
    GEMINI_API_KEY,
    MAX_DURATION_SEC,
    MIN_DURATION_SEC,
    MIN_FRAMES,
    PER_VIDEO_CONTEXT,
    RAW_VIDEO_DIR,
    TARGET_FPS,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TRIGGER_TOKEN,
    VIDEO_DIR,
)
from src.data.analyze import analyze_all
from src.data.preview import preview_videos
from src.data.process import process_all
from src.captioning.gemini import caption_all
from src.training.validate import validate_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare HunyuanVideo training dataset")
    parser.add_argument("--skip-caption", action="store_true", help="Skip captioning step")
    parser.add_argument("--skip-preview", action="store_true", help="Skip thumbnail preview")
    parser.add_argument("--raw-dir", type=str, default=None, help="Override raw video directory")
    args = parser.parse_args()

    raw_dir = args.raw_dir or str(RAW_VIDEO_DIR)
    dataset_dir = str(DATASET_DIR)
    video_dir = str(VIDEO_DIR)

    # Ensure output dirs exist
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Analyze raw videos
    print(f"\n{'=' * 60}")
    print("STEP 1: Analyzing raw videos")
    print(f"{'=' * 60}\n")
    analysis = analyze_all(raw_dir, min_duration=MIN_DURATION_SEC)
    if not analysis:
        print(f"No valid videos found in {raw_dir}")
        sys.exit(1)

    # Step 2: Process videos
    print(f"\n{'=' * 60}")
    print("STEP 2: Processing videos (resize, crop, trim)")
    print(f"{'=' * 60}\n")
    processed = process_all(
        video_analysis=analysis,
        output_dir=video_dir,
        target_w=TARGET_WIDTH,
        target_h=TARGET_HEIGHT,
        target_fps=TARGET_FPS,
        max_duration=MAX_DURATION_SEC,
        min_duration=MIN_DURATION_SEC,
        min_frames=MIN_FRAMES,
    )
    if not processed:
        print("No videos survived processing.")
        sys.exit(1)

    # Step 3: Preview
    if not args.skip_preview:
        print(f"\n{'=' * 60}")
        print("STEP 3: Preview thumbnails")
        print(f"{'=' * 60}\n")
        preview_videos(processed, TARGET_WIDTH, TARGET_HEIGHT, TARGET_FPS)
    else:
        print("\nSkipping preview.")

    # Step 4: Caption with Gemini
    if not args.skip_caption:
        print(f"\n{'=' * 60}")
        print("STEP 4: Auto-captioning with Gemini")
        print(f"{'=' * 60}\n")
        if not GEMINI_API_KEY:
            print("ERROR: GEMINI_API_KEY not set. Add it to .env file.")
            sys.exit(1)
        caption_all(
            video_dir=video_dir,
            dataset_dir=dataset_dir,
            trigger_token=TRIGGER_TOKEN,
            context=CAPTION_CONTEXT,
            api_key=GEMINI_API_KEY,
            per_video_context=PER_VIDEO_CONTEXT,
        )
    else:
        print("\nSkipping captioning (using existing captions).")

    # Step 5: Validate
    print(f"\n{'=' * 60}")
    print("STEP 5: Validating dataset")
    print(f"{'=' * 60}\n")
    is_valid = validate_dataset(
        dataset_dir=dataset_dir,
        target_w=TARGET_WIDTH,
        target_h=TARGET_HEIGHT,
        min_frames=MIN_FRAMES,
    )
    if not is_valid:
        print("\nDataset has errors. Fix them before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
