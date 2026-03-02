"""Tests for src.training.validate — dataset validation."""

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def _make_dataset(tmp_path):
    """Helper to create a dataset directory with videos.txt and prompts.txt."""

    def _inner(videos, prompts, create_files=True, file_size=200_000):
        (tmp_path / "videos.txt").write_text("\n".join(videos))
        (tmp_path / "prompts.txt").write_text("\n".join(prompts))
        if create_files:
            for vpath in videos:
                os.makedirs(os.path.dirname(vpath), exist_ok=True)
                with open(vpath, "wb") as f:
                    f.write(b"\x00" * file_size)
        return str(tmp_path)

    return _inner


# ── validate_dataset ─────────────────────────────────────────────────


@patch("src.training.validate.analyze_video")
def test_validate_passes_good_dataset(_mock_analyze, _make_dataset):
    """Valid dataset returns True."""
    _mock_analyze.return_value = {
        "width": 768,
        "height": 512,
        "frame_count": 120,
    }
    ds = _make_dataset(
        ["/tmp/test_vid/a.mp4", "/tmp/test_vid/b.mp4"],
        ["ohwx person on beach with sunset", "ohwx person dancing in garden"],
    )

    from src.training.validate import validate_dataset

    assert validate_dataset(ds, 768, 512) is True


@patch("src.training.validate.analyze_video")
def test_validate_mismatched_counts(_mock_analyze, _make_dataset):
    """Mismatched video/prompt counts returns False."""
    _mock_analyze.return_value = {
        "width": 768,
        "height": 512,
        "frame_count": 120,
    }
    ds = _make_dataset(
        ["/tmp/test_vid/a.mp4"],
        ["caption one", "caption two"],
        create_files=True,
    )

    from src.training.validate import validate_dataset

    assert validate_dataset(ds, 768, 512) is False


@patch("src.training.validate.analyze_video")
def test_validate_wrong_resolution(_mock_analyze, _make_dataset):
    """Wrong resolution fails validation."""
    _mock_analyze.return_value = {
        "width": 1920,
        "height": 1080,
        "frame_count": 120,
    }
    ds = _make_dataset(
        ["/tmp/test_vid/a.mp4"],
        ["ohwx person on a sandy beach at dusk"],
    )

    from src.training.validate import validate_dataset

    assert validate_dataset(ds, 768, 512) is False


@patch("src.training.validate.analyze_video")
def test_validate_too_few_frames(_mock_analyze, _make_dataset):
    """Too few frames fails validation."""
    _mock_analyze.return_value = {
        "width": 768,
        "height": 512,
        "frame_count": 5,
    }
    ds = _make_dataset(
        ["/tmp/test_vid/a.mp4"],
        ["ohwx person walking down a city street"],
    )

    from src.training.validate import validate_dataset

    assert validate_dataset(ds, 768, 512) is False


@patch("src.training.validate.analyze_video")
def test_validate_bad_caption(_mock_analyze, _make_dataset):
    """[FAILED] caption fails validation."""
    _mock_analyze.return_value = {
        "width": 768,
        "height": 512,
        "frame_count": 120,
    }
    ds = _make_dataset(
        ["/tmp/test_vid/a.mp4"],
        ["[FAILED] Could not extract frame"],
    )

    from src.training.validate import validate_dataset

    assert validate_dataset(ds, 768, 512) is False


def test_validate_missing_video_file(_make_dataset):
    """Missing video file fails validation."""
    ds = _make_dataset(
        ["/nonexistent/path/video.mp4"],
        ["ohwx person smiling warmly at the camera"],
        create_files=False,
    )

    from src.training.validate import validate_dataset

    assert validate_dataset(ds, 768, 512) is False
