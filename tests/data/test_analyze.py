"""Tests for src.data.analyze — video metadata extraction."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import (
    CV2_CAP_PROP_FPS,
    CV2_CAP_PROP_FRAME_COUNT,
    CV2_CAP_PROP_FRAME_HEIGHT,
    CV2_CAP_PROP_FRAME_WIDTH,
    make_mock_video_capture,
)


# ── analyze_video ────────────────────────────────────────────────────


@patch("src.data.analyze.cv2")
def test_analyze_video_normal(mock_cv2):
    """Normal video returns correct metadata dict."""
    cap = make_mock_video_capture(width=1920, height=1080, fps=30.0, frame_count=150)
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.CAP_PROP_FRAME_WIDTH = CV2_CAP_PROP_FRAME_WIDTH
    mock_cv2.CAP_PROP_FRAME_HEIGHT = CV2_CAP_PROP_FRAME_HEIGHT
    mock_cv2.CAP_PROP_FPS = CV2_CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    mock_cv2.COLOR_BGR2GRAY = 6
    mock_cv2.cvtColor.return_value = np.full((1080, 1920), 128.0, dtype=np.uint8)

    from src.data.analyze import analyze_video

    info = analyze_video("/fake/video.mp4")

    assert info is not None
    assert info["width"] == 1920
    assert info["height"] == 1080
    assert info["fps"] == 30.0
    assert info["frame_count"] == 150
    assert info["duration_sec"] == 5.0
    assert info["aspect_ratio"] == 1.78
    assert not info["is_likely_black"]
    assert not info["is_likely_overexposed"]


@patch("src.data.analyze.cv2")
def test_analyze_video_cannot_open(mock_cv2):
    """Video that can't be opened returns None."""
    cap = make_mock_video_capture(opens=False)
    mock_cv2.VideoCapture.return_value = cap

    from src.data.analyze import analyze_video

    assert analyze_video("/fake/broken.mp4") is None


@patch("src.data.analyze.cv2")
def test_analyze_video_zero_fps(mock_cv2):
    """Zero FPS avoids division by zero; duration_sec == 0."""
    cap = make_mock_video_capture(fps=0.0, frame_count=100)
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.CAP_PROP_FRAME_WIDTH = CV2_CAP_PROP_FRAME_WIDTH
    mock_cv2.CAP_PROP_FRAME_HEIGHT = CV2_CAP_PROP_FRAME_HEIGHT
    mock_cv2.CAP_PROP_FPS = CV2_CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    mock_cv2.COLOR_BGR2GRAY = 6
    mock_cv2.cvtColor.return_value = np.full((1080, 1920), 128.0, dtype=np.uint8)

    from src.data.analyze import analyze_video

    info = analyze_video("/fake/zero_fps.mp4")
    assert info["duration_sec"] == 0


@patch("src.data.analyze.cv2")
def test_analyze_video_zero_height(mock_cv2):
    """Zero height avoids division by zero; aspect_ratio == 0."""
    cap = make_mock_video_capture(height=0, frame_count=100, read_succeeds=False)
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.CAP_PROP_FRAME_WIDTH = CV2_CAP_PROP_FRAME_WIDTH
    mock_cv2.CAP_PROP_FRAME_HEIGHT = CV2_CAP_PROP_FRAME_HEIGHT
    mock_cv2.CAP_PROP_FPS = CV2_CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = 1

    from src.data.analyze import analyze_video

    info = analyze_video("/fake/zero_h.mp4")
    assert info["aspect_ratio"] == 0


@patch("src.data.analyze.cv2")
def test_analyze_video_read_fails(mock_cv2):
    """Frame read failure defaults brightness to 0 and is_likely_black True."""
    cap = make_mock_video_capture(read_succeeds=False)
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.CAP_PROP_FRAME_WIDTH = CV2_CAP_PROP_FRAME_WIDTH
    mock_cv2.CAP_PROP_FRAME_HEIGHT = CV2_CAP_PROP_FRAME_HEIGHT
    mock_cv2.CAP_PROP_FPS = CV2_CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = 1

    from src.data.analyze import analyze_video

    info = analyze_video("/fake/no_read.mp4")
    assert info["mean_brightness"] == 0
    assert info["is_likely_black"] is True


@patch("src.data.analyze.cv2")
def test_analyze_video_black_frame(mock_cv2):
    """Very dark video flags is_likely_black."""
    cap = make_mock_video_capture(brightness=5.0)
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.CAP_PROP_FRAME_WIDTH = CV2_CAP_PROP_FRAME_WIDTH
    mock_cv2.CAP_PROP_FRAME_HEIGHT = CV2_CAP_PROP_FRAME_HEIGHT
    mock_cv2.CAP_PROP_FPS = CV2_CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    mock_cv2.COLOR_BGR2GRAY = 6
    mock_cv2.cvtColor.return_value = np.full((1080, 1920), 5.0, dtype=np.uint8)

    from src.data.analyze import analyze_video

    info = analyze_video("/fake/dark.mp4")
    assert info["is_likely_black"] == True


@patch("src.data.analyze.cv2")
def test_analyze_video_overexposed(mock_cv2):
    """Very bright video flags is_likely_overexposed."""
    cap = make_mock_video_capture(brightness=250.0)
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.CAP_PROP_FRAME_WIDTH = CV2_CAP_PROP_FRAME_WIDTH
    mock_cv2.CAP_PROP_FRAME_HEIGHT = CV2_CAP_PROP_FRAME_HEIGHT
    mock_cv2.CAP_PROP_FPS = CV2_CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    mock_cv2.COLOR_BGR2GRAY = 6
    mock_cv2.cvtColor.return_value = np.full((1080, 1920), 250.0, dtype=np.uint8)

    from src.data.analyze import analyze_video

    info = analyze_video("/fake/bright.mp4")
    assert info["is_likely_overexposed"] == True


# ── find_videos ──────────────────────────────────────────────────────


def test_find_videos_with_mixed_extensions(tmp_path):
    """Finds mp4, mov, avi files and ignores non-video files."""
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.MOV").touch()
    (tmp_path / "c.avi").touch()
    (tmp_path / "d.txt").touch()
    (tmp_path / "e.jpg").touch()

    from src.data.analyze import find_videos

    found = find_videos(str(tmp_path))
    basenames = [os.path.basename(f) for f in found]
    assert "a.mp4" in basenames
    assert "c.avi" in basenames
    assert "d.txt" not in basenames
    assert "e.jpg" not in basenames


def test_find_videos_empty_dir(tmp_path):
    """Empty directory returns empty list."""
    from src.data.analyze import find_videos

    assert find_videos(str(tmp_path)) == []


# ── analyze_all ──────────────────────────────────────────────────────


@patch("src.data.analyze.cv2")
def test_analyze_all_returns_valid_only(mock_cv2, tmp_path):
    """analyze_all returns only successfully analyzed videos."""
    (tmp_path / "good.mp4").touch()
    (tmp_path / "bad.mp4").touch()

    call_count = 0

    def side_effect(path):
        nonlocal call_count
        call_count += 1
        cap = MagicMock()
        if "bad" in path:
            cap.isOpened.return_value = False
        else:
            cap.isOpened.return_value = True

            def _get(pid):
                return {
                    CV2_CAP_PROP_FRAME_WIDTH: 1920.0,
                    CV2_CAP_PROP_FRAME_HEIGHT: 1080.0,
                    CV2_CAP_PROP_FPS: 30.0,
                    CV2_CAP_PROP_FRAME_COUNT: 150.0,
                }.get(pid, 0.0)

            cap.get.side_effect = _get
            cap.read.return_value = (False, None)
        return cap

    mock_cv2.VideoCapture.side_effect = side_effect
    mock_cv2.CAP_PROP_FRAME_WIDTH = CV2_CAP_PROP_FRAME_WIDTH
    mock_cv2.CAP_PROP_FRAME_HEIGHT = CV2_CAP_PROP_FRAME_HEIGHT
    mock_cv2.CAP_PROP_FPS = CV2_CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = 1

    from src.data.analyze import analyze_all

    results = analyze_all(str(tmp_path))
    assert len(results) == 1
    assert results[0]["filename"] == "good.mp4"


@patch("src.data.analyze.cv2")
def test_analyze_all_warns_short_videos(mock_cv2, tmp_path):
    """Short videos get included but warning is printed."""
    (tmp_path / "short.mp4").touch()

    cap = make_mock_video_capture(fps=30.0, frame_count=10)  # 0.33s
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.CAP_PROP_FRAME_WIDTH = CV2_CAP_PROP_FRAME_WIDTH
    mock_cv2.CAP_PROP_FRAME_HEIGHT = CV2_CAP_PROP_FRAME_HEIGHT
    mock_cv2.CAP_PROP_FPS = CV2_CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    mock_cv2.COLOR_BGR2GRAY = 6
    mock_cv2.cvtColor.return_value = np.full((1080, 1920), 128.0, dtype=np.uint8)

    from src.data.analyze import analyze_all

    results = analyze_all(str(tmp_path), min_duration=1.0)
    assert len(results) == 1
    assert results[0]["duration_sec"] < 1.0
