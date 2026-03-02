"""Tests for src.data.preview — thumbnail grid display."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import (
    CV2_CAP_PROP_FRAME_COUNT,
    CV2_CAP_PROP_POS_FRAMES,
    make_mock_video_capture,
)


# ── get_thumbnail_grid ───────────────────────────────────────────────


@patch("src.data.preview.cv2")
def test_get_thumbnail_grid_returns_4_frames(mock_cv2):
    """Default call returns 4 thumbnail frames."""
    cap = make_mock_video_capture(frame_count=100)
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = CV2_CAP_PROP_POS_FRAMES
    mock_cv2.COLOR_BGR2RGB = 4
    mock_cv2.cvtColor.return_value = np.zeros((512, 768, 3), dtype=np.uint8)

    from src.data.preview import get_thumbnail_grid

    frames = get_thumbnail_grid("/fake/video.mp4")
    assert len(frames) == 4


@patch("src.data.preview.cv2")
def test_get_thumbnail_grid_read_fails(mock_cv2):
    """If frame reads fail, returns fewer frames."""
    cap = make_mock_video_capture(frame_count=100, read_succeeds=False)
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.CAP_PROP_FRAME_COUNT = CV2_CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = CV2_CAP_PROP_POS_FRAMES

    from src.data.preview import get_thumbnail_grid

    frames = get_thumbnail_grid("/fake/video.mp4")
    assert len(frames) == 0


# ── preview_videos ───────────────────────────────────────────────────


@patch("src.data.preview.plt")
@patch("src.data.preview.get_thumbnail_grid")
def test_preview_videos_empty(mock_grid, mock_plt):
    """Empty list prints message and does not create figure."""
    from src.data.preview import preview_videos

    preview_videos([], 768, 512, 24)
    mock_plt.subplots.assert_not_called()


@patch("src.data.preview.plt")
@patch("src.data.preview.get_thumbnail_grid")
def test_preview_videos_creates_figure(mock_grid, mock_plt):
    """With videos, creates subplot grid and calls show()."""
    mock_grid.return_value = [np.zeros((512, 768, 3))] * 4

    # For n_videos==1, preview.py wraps axes in a list: axes = [axes]
    # subplots returns (fig, array_of_axes) where array_of_axes is 1-D for 1 row
    axes_row = [MagicMock() for _ in range(4)]
    mock_plt.subplots.return_value = (MagicMock(), axes_row)

    from src.data.preview import preview_videos

    videos = [{"path": "/fake/v.mp4", "original": "v.mp4"}]
    preview_videos(videos, 768, 512, 24)

    mock_plt.subplots.assert_called_once()
    mock_plt.show.assert_called_once()
