import os
import subprocess
from pathlib import Path

from tqdm import tqdm

from src.data.analyze import analyze_video


def process_video(
    input_path: str,
    output_path: str,
    target_w: int,
    target_h: int,
    target_fps: int,
    max_duration: float,
) -> bool:
    """Resize, crop, and trim a single video using ffmpeg."""
    vf_filters = (
        f"fps={target_fps},"
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h}"
    )
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-t", str(max_duration),
        "-vf", vf_filters,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FFmpeg error: {result.stderr[-200:]}")
        return False
    return True


def validate_processed_video(
    path: str,
    target_w: int,
    target_h: int,
    target_fps: int,
    min_frames: int = 17,
) -> tuple[bool, str | dict]:
    """Validate a processed video meets target specs."""
    info = analyze_video(path)
    if not info:
        return False, "Could not open"

    issues = []
    if info["width"] != target_w:
        issues.append(f"width {info['width']} != {target_w}")
    if info["height"] != target_h:
        issues.append(f"height {info['height']} != {target_h}")
    if abs(info["fps"] - target_fps) > 1:
        issues.append(f"fps {info['fps']} != {target_fps}")
    if info["frame_count"] < min_frames:
        issues.append(f"only {info['frame_count']} frames")
    if info["is_likely_black"]:
        issues.append("black frames detected")

    if issues:
        return False, ", ".join(issues)
    return True, info


def process_all(
    video_analysis: list[dict],
    output_dir: str,
    target_w: int,
    target_h: int,
    target_fps: int,
    max_duration: float,
    min_duration: float = 1.0,
    min_frames: int = 17,
) -> list[dict]:
    """Process all analyzed videos: resize, crop, trim, and validate."""
    os.makedirs(output_dir, exist_ok=True)

    processed = []
    skipped = []
    print(f"Processing {len(video_analysis)} videos...\n")

    for info in tqdm(video_analysis, desc="Processing"):
        filename = info["filename"]
        stem = Path(filename).stem

        if info["is_likely_black"]:
            skipped.append((filename, "black frames"))
            continue
        if info["duration_sec"] < min_duration:
            skipped.append((filename, f"too short ({info['duration_sec']}s)"))
            continue

        output_name = f"{stem}_processed.mp4"
        output_path = os.path.join(output_dir, output_name)

        success = process_video(
            info["path"], output_path, target_w, target_h, target_fps, max_duration
        )
        if success:
            valid, result = validate_processed_video(
                output_path, target_w, target_h, target_fps, min_frames
            )
            if valid:
                processed.append({
                    "original": filename,
                    "processed": output_name,
                    "path": output_path,
                    "info": result,
                })
            else:
                skipped.append((filename, f"validation failed: {result}"))
                os.remove(output_path)
        else:
            skipped.append((filename, "processing failed"))

    print(f"\n{'=' * 60}")
    print(f"Successfully processed: {len(processed)} videos")
    print(f"Skipped: {len(skipped)} videos")
    for name, reason in skipped:
        print(f"   - {name}: {reason}")

    return processed
