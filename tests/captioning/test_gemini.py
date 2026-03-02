"""Tests for src.captioning.gemini — Gemini video captioning."""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import (
    CV2_CAP_PROP_FRAME_COUNT,
    CV2_CAP_PROP_POS_FRAMES,
    make_mock_video_capture,
)


# ── caption_video_with_gemini ────────────────────────────────────────


@patch("src.captioning.gemini.cv2")
def test_caption_success(mock_cv2):
    """Successful caption returns stripped text."""
    cap = make_mock_video_capture(frame_count=100)
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = CV2_CAP_PROP_POS_FRAMES
    mock_cv2.imencode.return_value = (True, MagicMock(tobytes=lambda: b"jpeg_bytes"))

    client = MagicMock()
    response = MagicMock()
    response.text = "  ohwx person walking on a beach.  "
    client.models.generate_content.return_value = response

    from src.captioning.gemini import caption_video_with_gemini

    result = caption_video_with_gemini("/fake/vid.mp4", "ohwx", "context", client)
    assert result == "ohwx person walking on a beach."
    client.models.generate_content.assert_called_once()


@patch("src.captioning.gemini.cv2")
def test_caption_frame_extraction_fails(mock_cv2):
    """Frame extraction failure returns None."""
    cap = make_mock_video_capture(frame_count=100, read_succeeds=False)
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = CV2_CAP_PROP_POS_FRAMES

    client = MagicMock()

    from src.captioning.gemini import caption_video_with_gemini

    result = caption_video_with_gemini("/fake/vid.mp4", "ohwx", "context", client)
    assert result is None
    client.models.generate_content.assert_not_called()


@patch("src.captioning.gemini.cv2")
def test_caption_api_exception(mock_cv2):
    """API exception propagates (not caught internally)."""
    cap = make_mock_video_capture(frame_count=100)
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = CV2_CAP_PROP_POS_FRAMES
    mock_cv2.imencode.return_value = (True, MagicMock(tobytes=lambda: b"jpeg"))

    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("API error")

    from src.captioning.gemini import caption_video_with_gemini

    with pytest.raises(RuntimeError, match="API error"):
        caption_video_with_gemini("/fake/vid.mp4", "ohwx", "context", client)


# ── caption_all ──────────────────────────────────────────────────────


@patch("src.captioning.gemini.time")
@patch("src.captioning.gemini.caption_video_with_gemini")
@patch("src.captioning.gemini.genai")
@patch("src.captioning.gemini.glob")
def test_caption_all_writes_output_files(mock_glob, mock_genai, mock_caption, mock_time, tmp_path):
    """caption_all writes captions.json, videos.txt, and prompts.txt."""
    video_dir = str(tmp_path / "videos")
    os.makedirs(video_dir)
    for name in ["a.mp4", "b.mp4"]:
        open(os.path.join(video_dir, name), "w").close()

    mock_glob.glob.return_value = [
        os.path.join(video_dir, "a.mp4"),
        os.path.join(video_dir, "b.mp4"),
    ]
    mock_caption.side_effect = ["ohwx person on beach", "ohwx person dancing"]
    mock_time.sleep = MagicMock()

    dataset_dir = str(tmp_path / "dataset")
    os.makedirs(dataset_dir)

    from src.captioning.gemini import caption_all

    result = caption_all(
        video_dir=video_dir,
        dataset_dir=dataset_dir,
        trigger_token="ohwx",
        context="test context",
        api_key="fake_key",
    )

    assert len(result) == 2
    assert os.path.exists(os.path.join(dataset_dir, "captions.json"))
    assert os.path.exists(os.path.join(dataset_dir, "videos.txt"))
    assert os.path.exists(os.path.join(dataset_dir, "prompts.txt"))

    with open(os.path.join(dataset_dir, "videos.txt")) as f:
        lines = f.read().strip().split("\n")
    assert len(lines) == 2


@patch("src.captioning.gemini.time")
@patch("src.captioning.gemini.caption_video_with_gemini")
@patch("src.captioning.gemini.genai")
@patch("src.captioning.gemini.glob")
def test_caption_all_excludes_failed(mock_glob, mock_genai, mock_caption, mock_time, tmp_path):
    """Failed captions excluded from videos.txt and prompts.txt."""
    video_dir = str(tmp_path / "videos")
    os.makedirs(video_dir)
    for name in ["ok.mp4", "fail.mp4"]:
        open(os.path.join(video_dir, name), "w").close()

    mock_glob.glob.return_value = [
        os.path.join(video_dir, "fail.mp4"),
        os.path.join(video_dir, "ok.mp4"),
    ]
    mock_caption.side_effect = [None, "ohwx person smiling"]
    mock_time.sleep = MagicMock()

    dataset_dir = str(tmp_path / "dataset")
    os.makedirs(dataset_dir)

    from src.captioning.gemini import caption_all

    result = caption_all(
        video_dir=video_dir,
        dataset_dir=dataset_dir,
        trigger_token="ohwx",
        context="ctx",
        api_key="key",
    )

    with open(os.path.join(dataset_dir, "videos.txt")) as f:
        video_lines = f.read().strip().split("\n")
    with open(os.path.join(dataset_dir, "prompts.txt")) as f:
        prompt_lines = f.read().strip().split("\n")

    assert len(video_lines) == 1
    assert len(prompt_lines) == 1
    assert "ok.mp4" in video_lines[0]


@patch("src.captioning.gemini.time")
@patch("src.captioning.gemini.caption_video_with_gemini")
@patch("src.captioning.gemini.genai")
@patch("src.captioning.gemini.glob")
def test_caption_all_handles_exception(mock_glob, mock_genai, mock_caption, mock_time, tmp_path):
    """API exception for one video doesn't stop captioning others."""
    video_dir = str(tmp_path / "videos")
    os.makedirs(video_dir)
    for name in ["err.mp4", "ok.mp4"]:
        open(os.path.join(video_dir, name), "w").close()

    mock_glob.glob.return_value = [
        os.path.join(video_dir, "err.mp4"),
        os.path.join(video_dir, "ok.mp4"),
    ]
    mock_caption.side_effect = [RuntimeError("API down"), "ohwx person"]
    mock_time.sleep = MagicMock()

    dataset_dir = str(tmp_path / "dataset")
    os.makedirs(dataset_dir)

    from src.captioning.gemini import caption_all

    result = caption_all(
        video_dir=video_dir,
        dataset_dir=dataset_dir,
        trigger_token="ohwx",
        context="ctx",
        api_key="key",
    )

    assert "[FAILED]" in result["err.mp4"]
    assert result["ok.mp4"] == "ohwx person"


@patch("src.captioning.gemini.genai")
@patch("src.captioning.gemini.glob")
def test_caption_all_no_videos_raises(mock_glob, mock_genai, tmp_path):
    """No videos found raises FileNotFoundError."""
    mock_glob.glob.return_value = []

    dataset_dir = str(tmp_path / "dataset")
    os.makedirs(dataset_dir)

    from src.captioning.gemini import caption_all

    with pytest.raises(FileNotFoundError):
        caption_all(
            video_dir="/empty",
            dataset_dir=dataset_dir,
            trigger_token="ohwx",
            context="ctx",
            api_key="key",
        )
