"""Shared fixtures for all tests.

Pre-mocks heavy/unavailable third-party modules (cv2, matplotlib, tqdm,
google.genai, diffusers, dotenv) in sys.modules so that source modules
under src/ can be imported even in minimal test environments.
"""

import json
import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Pre-mock missing third-party modules ─────────────────────────────
# This must happen before any src.* imports so that top-level imports
# like `import cv2` or `from google import genai` resolve to mocks.

_MOCK_MODULES = [
    "cv2",
    "matplotlib",
    "matplotlib.pyplot",
    "tqdm",
    "google",
    "google.genai",
    "dotenv",
    "diffusers",
    "diffusers.utils",
]

for _mod_name in _MOCK_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# Make `from dotenv import load_dotenv` work
sys.modules["dotenv"].load_dotenv = MagicMock()

# Make `from tqdm import tqdm` return a passthrough (identity iterable)
sys.modules["tqdm"].tqdm = lambda iterable, **kw: iterable

# Make `from diffusers.utils import export_to_video` return a mock
sys.modules["diffusers.utils"].export_to_video = MagicMock()

# Ensure `from google import genai` works
sys.modules["google"].genai = sys.modules["google.genai"]


# ── cv2 constants (so cv2 doesn't need to be installed) ──────────────
CV2_CAP_PROP_FRAME_WIDTH = 3
CV2_CAP_PROP_FRAME_HEIGHT = 4
CV2_CAP_PROP_FPS = 5
CV2_CAP_PROP_FRAME_COUNT = 7
CV2_CAP_PROP_POS_FRAMES = 1


def make_mock_video_capture(
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    frame_count: int = 150,
    opens: bool = True,
    read_succeeds: bool = True,
    brightness: float = 128.0,
):
    """Factory that returns a mock cv2.VideoCapture with configurable props."""
    cap = MagicMock()
    cap.isOpened.return_value = opens

    def _get(prop_id):
        return {
            CV2_CAP_PROP_FRAME_WIDTH: float(width),
            CV2_CAP_PROP_FRAME_HEIGHT: float(height),
            CV2_CAP_PROP_FPS: fps,
            CV2_CAP_PROP_FRAME_COUNT: float(frame_count),
        }.get(prop_id, 0.0)

    cap.get.side_effect = _get

    # Fake grayscale frame for brightness calculations
    gray_frame = np.full((height, width), brightness, dtype=np.uint8)
    bgr_frame = np.stack([gray_frame] * 3, axis=-1)

    cap.read.return_value = (read_succeeds, bgr_frame if read_succeeds else None)
    return cap


@pytest.fixture
def mock_video_capture():
    """Provide the factory function as a fixture."""
    return make_mock_video_capture


@pytest.fixture
def sample_video_info():
    """A typical video analysis dict."""
    return {
        "path": "/fake/video.mp4",
        "filename": "video.mp4",
        "width": 768,
        "height": 512,
        "fps": 24.0,
        "frame_count": 120,
        "duration_sec": 5.0,
        "aspect_ratio": 1.5,
        "mean_brightness": 128.0,
        "std_brightness": 40.0,
        "is_likely_black": False,
        "is_likely_overexposed": False,
    }


@pytest.fixture
def sample_dataset_dir(tmp_path):
    """Create a temporary dataset directory with videos.txt and prompts.txt."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    # Create dummy video files
    for i in range(3):
        (video_dir / f"vid{i}_processed.mp4").write_bytes(b"\x00" * 1024)

    videos = [str(video_dir / f"vid{i}_processed.mp4") for i in range(3)]
    prompts = [
        "ohwx person walking on a beach at sunset",
        "ohwx person smiling in a garden with flowers",
        "ohwx person dancing under city lights at night",
    ]

    (tmp_path / "videos.txt").write_text("\n".join(videos))
    (tmp_path / "prompts.txt").write_text("\n".join(prompts))
    (tmp_path / "captions.json").write_text(
        json.dumps(dict(zip([f"vid{i}_processed.mp4" for i in range(3)], prompts)))
    )

    return tmp_path
