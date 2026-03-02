import cv2
import matplotlib.pyplot as plt


def get_thumbnail_grid(video_path: str, num_thumbs: int = 4) -> list:
    """Extract evenly-spaced thumbnail frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(num_thumbs):
        pos = int((i / num_thumbs) * total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def preview_videos(
    processed_videos: list[dict],
    target_w: int,
    target_h: int,
    target_fps: int,
) -> None:
    """Display a grid of thumbnails for all processed videos."""
    n_videos = len(processed_videos)
    if n_videos == 0:
        print("No processed videos to preview.")
        return

    fig, axes = plt.subplots(n_videos, 4, figsize=(16, 3 * n_videos))
    if n_videos == 1:
        axes = [axes]

    for row, vid in enumerate(processed_videos):
        thumbs = get_thumbnail_grid(vid["path"])
        for col, thumb in enumerate(thumbs):
            axes[row][col].imshow(thumb)
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_title(vid["original"], fontsize=10, loc="left")

    plt.suptitle(
        f"Processed Videos ({target_w}x{target_h} @ {target_fps}fps)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
