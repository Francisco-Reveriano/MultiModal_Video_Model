"""Tests for src.training.gpu_utils — GPU capability detection."""

from unittest.mock import MagicMock, patch

import torch
import pytest


# ── get_compute_capability ───────────────────────────────────────────


@patch("src.training.gpu_utils.torch")
def test_compute_capability_with_cuda(mock_torch):
    """Returns (major, minor) when CUDA available."""
    mock_torch.cuda.is_available.return_value = True
    props = MagicMock()
    props.major = 8
    props.minor = 9
    mock_torch.cuda.get_device_properties.return_value = props

    from src.training.gpu_utils import get_compute_capability

    assert get_compute_capability() == (8, 9)


@patch("src.training.gpu_utils.torch")
def test_compute_capability_no_cuda(mock_torch):
    """Returns None when no CUDA."""
    mock_torch.cuda.is_available.return_value = False

    from src.training.gpu_utils import get_compute_capability

    assert get_compute_capability() is None


# ── is_bf16_supported ────────────────────────────────────────────────


@patch("src.training.gpu_utils.torch")
def test_bf16_supported(mock_torch):
    """True when GPU supports bf16."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.is_bf16_supported.return_value = True

    from src.training.gpu_utils import is_bf16_supported

    assert is_bf16_supported() is True


@patch("src.training.gpu_utils.torch")
def test_bf16_unsupported(mock_torch):
    """False when GPU doesn't support bf16."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.is_bf16_supported.return_value = False

    from src.training.gpu_utils import is_bf16_supported

    assert is_bf16_supported() is False


@patch("src.training.gpu_utils.torch")
def test_bf16_no_cuda(mock_torch):
    """False when no CUDA at all."""
    mock_torch.cuda.is_available.return_value = False

    from src.training.gpu_utils import is_bf16_supported

    assert is_bf16_supported() is False


# ── is_fp8_supported ────────────────────────────────────────────────


@patch("src.training.gpu_utils.torch")
def test_fp8_supported_sm89(mock_torch):
    """True for sm_89 (Ada Lovelace)."""
    mock_torch.cuda.is_available.return_value = True
    props = MagicMock()
    props.major = 8
    props.minor = 9
    mock_torch.cuda.get_device_properties.return_value = props

    from src.training.gpu_utils import is_fp8_supported

    assert is_fp8_supported() is True


@patch("src.training.gpu_utils.torch")
def test_fp8_supported_sm90(mock_torch):
    """True for sm_90 (Hopper)."""
    mock_torch.cuda.is_available.return_value = True
    props = MagicMock()
    props.major = 9
    props.minor = 0
    mock_torch.cuda.get_device_properties.return_value = props

    from src.training.gpu_utils import is_fp8_supported

    assert is_fp8_supported() is True


@patch("src.training.gpu_utils.torch")
def test_fp8_unsupported_sm80(mock_torch):
    """False for sm_80 (Ampere A100)."""
    mock_torch.cuda.is_available.return_value = True
    props = MagicMock()
    props.major = 8
    props.minor = 0
    mock_torch.cuda.get_device_properties.return_value = props

    from src.training.gpu_utils import is_fp8_supported

    assert is_fp8_supported() is False


@patch("src.training.gpu_utils.torch")
def test_fp8_no_cuda(mock_torch):
    """False when no CUDA."""
    mock_torch.cuda.is_available.return_value = False

    from src.training.gpu_utils import is_fp8_supported

    assert is_fp8_supported() is False


# ── get_supported_dtype ──────────────────────────────────────────────


@patch("src.training.gpu_utils.is_bf16_supported")
def test_dtype_bf16(mock_bf16):
    """Returns 'bf16' when bf16 supported."""
    mock_bf16.return_value = True

    from src.training.gpu_utils import get_supported_dtype

    assert get_supported_dtype() == "bf16"


@patch("src.training.gpu_utils.is_bf16_supported")
def test_dtype_fp16_fallback(mock_bf16):
    """Returns 'fp16' when bf16 not supported."""
    mock_bf16.return_value = False

    from src.training.gpu_utils import get_supported_dtype

    assert get_supported_dtype() == "fp16"


# ── get_supported_torch_dtype ────────────────────────────────────────


@patch("src.training.gpu_utils.is_bf16_supported")
def test_torch_dtype_bfloat16(mock_bf16):
    """Returns torch.bfloat16 when bf16 supported."""
    mock_bf16.return_value = True

    from src.training.gpu_utils import get_supported_torch_dtype

    assert get_supported_torch_dtype() == torch.bfloat16


@patch("src.training.gpu_utils.is_bf16_supported")
def test_torch_dtype_float16_fallback(mock_bf16):
    """Returns torch.float16 when bf16 not supported."""
    mock_bf16.return_value = False

    from src.training.gpu_utils import get_supported_torch_dtype

    assert get_supported_torch_dtype() == torch.float16
