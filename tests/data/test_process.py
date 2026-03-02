"""Tests for src.data.process — video processing with ffmpeg."""

import os
from unittest.mock import MagicMock, patch

import pytest


# ── process_video ────────────────────────────────────────────────────


@patch("src.data.process.subprocess")
def test_process_video_success(mock_subprocess, tmp_path):
    """Successful ffmpeg call returns True."""
    mock_subprocess.run.return_value = MagicMock(returncode=0)

    from src.data.process import process_video

    result = process_video(
        input_path="/fake/in.mp4",
        output_path=str(tmp_path / "out.mp4"),
        target_w=768,
        target_h=512,
        target_fps=24,
        max_duration=5.0,
    )
    assert result is True
    mock_subprocess.run.assert_called_once()
    cmd = mock_subprocess.run.call_args[0][0]
    assert "ffmpeg" in cmd
    assert "-t" in cmd
    assert "5.0" in cmd


@patch("src.data.process.subprocess")
def test_process_video_ffmpeg_failure(mock_subprocess, tmp_path):
    """Non-zero ffmpeg exit returns False."""
    mock_subprocess.run.return_value = MagicMock(
        returncode=1, stderr="Error encoding"
    )

    from src.data.process import process_video

    result = process_video(
        input_path="/fake/in.mp4",
        output_path=str(tmp_path / "out.mp4"),
        target_w=768,
        target_h=512,
        target_fps=24,
        max_duration=5.0,
    )
    assert result is False


@patch("src.data.process.subprocess")
def test_process_video_cmd_contains_resolution(mock_subprocess, tmp_path):
    """ffmpeg command contains the target resolution in the scale filter."""
    mock_subprocess.run.return_value = MagicMock(returncode=0)

    from src.data.process import process_video

    process_video("/fake/in.mp4", str(tmp_path / "out.mp4"), 768, 512, 24, 5.0)
    cmd = mock_subprocess.run.call_args[0][0]
    vf_idx = cmd.index("-vf")
    vf_filter = cmd[vf_idx + 1]
    assert "768" in vf_filter
    assert "512" in vf_filter


# ── validate_processed_video ─────────────────────────────────────────


@patch("src.data.process.analyze_video")
def test_validate_correct_video(mock_analyze):
    """Video matching target specs passes validation."""
    mock_analyze.return_value = {
        "width": 768,
        "height": 512,
        "fps": 24.0,
        "frame_count": 120,
        "is_likely_black": False,
    }

    from src.data.process import validate_processed_video

    valid, result = validate_processed_video("/fake/ok.mp4", 768, 512, 24)
    assert valid is True
    assert isinstance(result, dict)


@patch("src.data.process.analyze_video")
def test_validate_wrong_resolution(mock_analyze):
    """Wrong resolution fails validation."""
    mock_analyze.return_value = {
        "width": 1920,
        "height": 1080,
        "fps": 24.0,
        "frame_count": 120,
        "is_likely_black": False,
    }

    from src.data.process import validate_processed_video

    valid, msg = validate_processed_video("/fake/bad.mp4", 768, 512, 24)
    assert valid is False
    assert "width" in msg


@patch("src.data.process.analyze_video")
def test_validate_too_few_frames(mock_analyze):
    """Too few frames fails validation."""
    mock_analyze.return_value = {
        "width": 768,
        "height": 512,
        "fps": 24.0,
        "frame_count": 10,
        "is_likely_black": False,
    }

    from src.data.process import validate_processed_video

    valid, msg = validate_processed_video("/fake/short.mp4", 768, 512, 24)
    assert valid is False
    assert "frames" in msg


@patch("src.data.process.analyze_video")
def test_validate_black_frames(mock_analyze):
    """Black frames detected fails validation."""
    mock_analyze.return_value = {
        "width": 768,
        "height": 512,
        "fps": 24.0,
        "frame_count": 120,
        "is_likely_black": True,
    }

    from src.data.process import validate_processed_video

    valid, msg = validate_processed_video("/fake/black.mp4", 768, 512, 24)
    assert valid is False
    assert "black" in msg


@patch("src.data.process.analyze_video")
def test_validate_cannot_open(mock_analyze):
    """Video that can't be opened fails validation."""
    mock_analyze.return_value = None

    from src.data.process import validate_processed_video

    valid, msg = validate_processed_video("/fake/broken.mp4", 768, 512, 24)
    assert valid is False
    assert "Could not open" in msg


# ── process_all ──────────────────────────────────────────────────────


@patch("src.data.process.validate_processed_video")
@patch("src.data.process.process_video")
def test_process_all_skips_black_videos(mock_process, mock_validate, tmp_path):
    """Black-flagged videos are skipped entirely."""
    analysis = [
        {
            "filename": "good.mp4",
            "path": "/fake/good.mp4",
            "is_likely_black": False,
            "duration_sec": 3.0,
        },
        {
            "filename": "black.mp4",
            "path": "/fake/black.mp4",
            "is_likely_black": True,
            "duration_sec": 3.0,
        },
    ]
    mock_process.return_value = True
    mock_validate.return_value = (
        True,
        {"width": 768, "height": 512, "fps": 24, "frame_count": 72},
    )

    from src.data.process import process_all

    results = process_all(analysis, str(tmp_path), 768, 512, 24, 5.0)
    assert len(results) == 1
    assert results[0]["original"] == "good.mp4"


@patch("src.data.process.validate_processed_video")
@patch("src.data.process.process_video")
def test_process_all_skips_short_videos(mock_process, mock_validate, tmp_path):
    """Too-short videos are skipped."""
    analysis = [
        {
            "filename": "short.mp4",
            "path": "/fake/short.mp4",
            "is_likely_black": False,
            "duration_sec": 0.3,
        },
    ]

    from src.data.process import process_all

    results = process_all(analysis, str(tmp_path), 768, 512, 24, 5.0, min_duration=1.0)
    assert len(results) == 0
    mock_process.assert_not_called()


@patch("src.data.process.os.remove")
@patch("src.data.process.validate_processed_video")
@patch("src.data.process.process_video")
def test_process_all_removes_failed_validation(
    mock_process, mock_validate, mock_remove, tmp_path
):
    """Failed validation triggers os.remove on the output file."""
    analysis = [
        {
            "filename": "bad.mp4",
            "path": "/fake/bad.mp4",
            "is_likely_black": False,
            "duration_sec": 3.0,
        },
    ]
    mock_process.return_value = True
    mock_validate.return_value = (False, "wrong resolution")

    from src.data.process import process_all

    results = process_all(analysis, str(tmp_path), 768, 512, 24, 5.0)
    assert len(results) == 0
    mock_remove.assert_called_once()


@patch("src.data.process.validate_processed_video")
@patch("src.data.process.process_video")
def test_process_all_handles_processing_failure(mock_process, mock_validate, tmp_path):
    """Processing failure is recorded as skipped."""
    analysis = [
        {
            "filename": "fail.mp4",
            "path": "/fake/fail.mp4",
            "is_likely_black": False,
            "duration_sec": 3.0,
        },
    ]
    mock_process.return_value = False

    from src.data.process import process_all

    results = process_all(analysis, str(tmp_path), 768, 512, 24, 5.0)
    assert len(results) == 0
    mock_validate.assert_not_called()
