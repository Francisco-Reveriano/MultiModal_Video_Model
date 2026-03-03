import os
import subprocess
from pathlib import Path

from src.data.analyze import analyze_video


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-500:])


def _video_duration_sec(video_path: str) -> float:
    info = analyze_video(video_path)
    if not info:
        raise RuntimeError(f"Could not analyze video: {video_path}")
    return float(info["duration_sec"])


def _temporal_start_times(duration_sec: float, clip_duration_sec: float, num_crops: int) -> list[float]:
    if num_crops <= 0:
        return []
    max_start = max(0.0, duration_sec - clip_duration_sec)
    if max_start <= 0:
        return [0.0]
    if num_crops == 1:
        return [max_start / 2.0]
    return [round((i / (num_crops - 1)) * max_start, 3) for i in range(num_crops)]


def _temporal_crop(input_path: str, output_path: str, start_sec: float, duration_sec: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-i",
        input_path,
        "-t",
        str(duration_sec),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        output_path,
    ]
    _run_ffmpeg(cmd)


def _horizontal_flip(input_path: str, output_path: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        "hflip",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        output_path,
    ]
    _run_ffmpeg(cmd)


def augment_processed_videos(
    processed_videos: list[dict],
    output_dir: str,
    temporal_crop_duration_sec: float = 3.0,
    temporal_crops_per_video: int = 2,
    include_horizontal_flip: bool = True,
) -> list[dict]:
    """
    Build an augmented dataset from already-processed videos.

    Returns records with at least:
    - filename
    - path
    - source
    - augmentation
    - is_flipped
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for video in processed_videos:
        src_path = video["path"]
        src_name = Path(src_path).name
        src_stem = Path(src_path).stem

        # Base sample (points to existing processed clip)
        base_record = {
            "filename": src_name,
            "path": src_path,
            "source": video["original"],
            "augmentation": "base",
            "is_flipped": False,
            "temporal_start_sec": None,
            "temporal_duration_sec": None,
        }
        records.append(base_record)

        # Temporal crops from base
        duration_sec = _video_duration_sec(src_path)
        starts = _temporal_start_times(duration_sec, temporal_crop_duration_sec, temporal_crops_per_video)
        crop_records = []
        for crop_idx, start_sec in enumerate(starts, start=1):
            crop_filename = f"{src_stem}_tc{crop_idx}.mp4"
            crop_path = str(output_dir_path / crop_filename)
            _temporal_crop(src_path, crop_path, start_sec, temporal_crop_duration_sec)
            crop_record = {
                "filename": crop_filename,
                "path": crop_path,
                "source": video["original"],
                "augmentation": "temporal_crop",
                "is_flipped": False,
                "temporal_start_sec": start_sec,
                "temporal_duration_sec": temporal_crop_duration_sec,
            }
            crop_records.append(crop_record)
        records.extend(crop_records)

        if include_horizontal_flip:
            # Flip base
            flip_base_filename = f"{src_stem}_flip.mp4"
            flip_base_path = str(output_dir_path / flip_base_filename)
            _horizontal_flip(src_path, flip_base_path)
            records.append(
                {
                    "filename": flip_base_filename,
                    "path": flip_base_path,
                    "source": video["original"],
                    "augmentation": "horizontal_flip",
                    "is_flipped": True,
                    "temporal_start_sec": None,
                    "temporal_duration_sec": None,
                }
            )

            # Flip each temporal crop
            for crop_record in crop_records:
                crop_stem = Path(crop_record["filename"]).stem
                flip_crop_filename = f"{crop_stem}_flip.mp4"
                flip_crop_path = str(output_dir_path / flip_crop_filename)
                _horizontal_flip(crop_record["path"], flip_crop_path)
                records.append(
                    {
                        "filename": flip_crop_filename,
                        "path": flip_crop_path,
                        "source": video["original"],
                        "augmentation": "temporal_crop+horizontal_flip",
                        "is_flipped": True,
                        "temporal_start_sec": crop_record["temporal_start_sec"],
                        "temporal_duration_sec": crop_record["temporal_duration_sec"],
                    }
                )

    return records
