"""Centralized GPU capability detection for training and inference."""

import torch


def get_compute_capability() -> tuple[int, int] | None:
    """Return (major, minor) compute capability of GPU 0, or None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor)


def is_bf16_supported() -> bool:
    """Check if the current GPU supports bfloat16."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.is_bf16_supported()


def is_fp8_supported() -> bool:
    """Check if the current GPU supports FP8 (requires sm_89+: Hopper/Ada)."""
    cc = get_compute_capability()
    if cc is None:
        return False
    return cc >= (8, 9)


def get_supported_dtype() -> str:
    """Return the best mixed-precision dtype string for accelerate: 'bf16' or 'fp16'."""
    return "bf16" if is_bf16_supported() else "fp16"


def get_supported_torch_dtype() -> torch.dtype:
    """Return the best torch dtype: bfloat16 if supported, else float16."""
    return torch.bfloat16 if is_bf16_supported() else torch.float16
