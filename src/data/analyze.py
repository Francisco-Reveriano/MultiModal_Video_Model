import glob
import os

import cv2


def analyze_video(path: str) -> dict | None:
    """Extract metadata from a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    info = {
        "path": path,
        "filename": os.path.basename(path),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = (
        round(info["frame_count"] / info["fps"], 2) if info["fps"] > 0 else 0
    )
    info["aspect_ratio"] = (
        round(info["width"] / info["height"], 2) if info["height"] > 0 else 0
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, info["frame_count"] // 2)
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info["mean_brightness"] = round(gray.mean(), 1)
        info["std_brightness"] = round(gray.std(), 1)
        info["is_likely_black"] = gray.mean() < 15
        info["is_likely_overexposed"] = gray.mean() > 240
    else:
        info["mean_brightness"] = 0
        info["std_brightness"] = 0
        info["is_likely_black"] = True
        info["is_likely_overexposed"] = False

    cap.release()
    return info


def find_videos(directory: str) -> list[str]:
    """Find all video files in a directory."""
    extensions = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi")
    videos = []
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(videos)


def analyze_all(video_dir: str, min_duration: float = 1.0) -> list[dict]:
    """Analyze all videos in a directory and print a summary."""
    raw_videos = find_videos(video_dir)
    print(f"Found {len(raw_videos)} video files\n")

    results = []
    for v in raw_videos:
        info = analyze_video(v)
        if info:
            results.append(info)
            status = "OK"
            warnings = []
            if info["duration_sec"] < min_duration:
                warnings.append("TOO SHORT")
                status = "WARN"
            if info["is_likely_black"]:
                warnings.append("BLACK FRAME")
                status = "FAIL"
            if info["is_likely_overexposed"]:
                warnings.append("OVEREXPOSED")
                status = "WARN"
            if info["duration_sec"] > 60:
                warnings.append("VERY LONG - will be trimmed")
            warn_str = f" [{', '.join(warnings)}]" if warnings else ""
            print(
                f"  [{status}] {info['filename']}: {info['width']}x{info['height']} "
                f"@ {info['fps']}fps, {info['duration_sec']}s, "
                f"brightness={info['mean_brightness']}{warn_str}"
            )
        else:
            print(f"  [FAIL] {os.path.basename(v)}: COULD NOT OPEN")

    print(f"\n{len(results)} valid videos analyzed")
    return results
