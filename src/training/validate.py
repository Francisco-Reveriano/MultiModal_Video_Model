import os

from src.data.analyze import analyze_video


def validate_dataset(
    dataset_dir: str,
    target_w: int,
    target_h: int,
    min_frames: int = 17,
) -> bool:
    """Run final validation on the prepared dataset."""
    print("Running final validation...\n")

    errors = []
    warnings = []

    videos_path = os.path.join(dataset_dir, "videos.txt")
    prompts_path = os.path.join(dataset_dir, "prompts.txt")

    with open(videos_path) as f:
        videos_txt = f.read().strip().split("\n")
    with open(prompts_path) as f:
        prompts_txt = f.read().strip().split("\n")

    if len(videos_txt) != len(prompts_txt):
        errors.append(
            f"Mismatch: {len(videos_txt)} videos vs {len(prompts_txt)} prompts"
        )
    if len(videos_txt) < 2:
        warnings.append(
            f"Only {len(videos_txt)} videos -- consider adding more (10+ recommended)"
        )

    for i, (vpath, caption) in enumerate(zip(videos_txt, prompts_txt)):
        if not os.path.exists(vpath):
            errors.append(f"Video not found: {vpath}")
            continue

        info = analyze_video(vpath)
        if info is None:
            errors.append(f"Video {i}: could not open")
            continue

        if info["width"] != target_w or info["height"] != target_h:
            errors.append(
                f"Video {i}: wrong resolution {info['width']}x{info['height']}"
            )
        if info["frame_count"] < min_frames:
            errors.append(
                f"Video {i}: only {info['frame_count']} frames (min {min_frames})"
            )
        if caption.startswith("[REPLACE]") or caption.startswith("[FAILED]"):
            errors.append(f"Video {i}: bad caption!")
        if len(caption) < 20:
            warnings.append(f"Video {i}: caption very short ({len(caption)} chars)")

        size_mb = os.path.getsize(vpath) / 1e6
        if size_mb < 0.1:
            errors.append(f"Video {i}: suspiciously small ({size_mb:.2f} MB)")

    if errors:
        print("ERRORS (must fix before training):")
        for e in errors:
            print(f"   - {e}")
    else:
        print("No errors found!")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w}")

    if not errors:
        print(f"\n{'=' * 60}")
        print("Dataset is ready for training!")
        print(f"   Path: {dataset_dir}")
        print(f"   Videos: {len(videos_txt)}")

    return len(errors) == 0
