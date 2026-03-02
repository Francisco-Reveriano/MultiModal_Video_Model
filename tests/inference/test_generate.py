"""Tests for src.inference.generate — pipeline loading and video generation."""

from unittest.mock import MagicMock, patch, call

import torch
import pytest


# ── load_pipeline ────────────────────────────────────────────────────


@patch("src.inference.generate.get_supported_torch_dtype")
@patch("src.inference.generate.torch")
def test_load_pipeline_no_cuda_raises(mock_torch, mock_dtype):
    """Raises RuntimeError when CUDA not available."""
    mock_torch.cuda.is_available.return_value = False

    from src.inference.generate import load_pipeline

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        load_pipeline("model_id", "/lora/path")


@patch("src.inference.generate.HunyuanVideoPipeline")
@patch("src.inference.generate.HunyuanVideoTransformer3DModel")
@patch("src.inference.generate.get_supported_torch_dtype")
@patch("src.inference.generate.torch")
def test_load_pipeline_uses_dynamic_dtype(
    mock_torch, mock_dtype, mock_transformer_cls, mock_pipe_cls
):
    """Pipeline loads transformer with dtype from gpu_utils."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.float16 = torch.float16
    mock_dtype.return_value = torch.bfloat16

    mock_transformer = MagicMock()
    mock_transformer_cls.from_pretrained.return_value = mock_transformer

    mock_pipe = MagicMock()
    mock_pipe_cls.from_pretrained.return_value = mock_pipe

    from src.inference.generate import load_pipeline

    pipe = load_pipeline("model_id", "/lora/path", lora_strength=0.7)

    mock_transformer_cls.from_pretrained.assert_called_once_with(
        "model_id", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    mock_pipe.load_lora_weights.assert_called_once_with(
        "/lora/path", adapter_name="hunyuan-lora"
    )
    mock_pipe.set_adapters.assert_called_once_with(["hunyuan-lora"], [0.7])


@patch("src.inference.generate.HunyuanVideoPipeline")
@patch("src.inference.generate.HunyuanVideoTransformer3DModel")
@patch("src.inference.generate.get_supported_torch_dtype")
@patch("src.inference.generate.torch")
def test_load_pipeline_enables_tiling_and_offload(
    mock_torch, mock_dtype, mock_transformer_cls, mock_pipe_cls
):
    """Pipeline enables VAE tiling and CPU offload."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.float16 = torch.float16
    mock_dtype.return_value = torch.float16

    mock_transformer = MagicMock()
    mock_transformer_cls.from_pretrained.return_value = mock_transformer

    mock_pipe = MagicMock()
    mock_pipe_cls.from_pretrained.return_value = mock_pipe

    from src.inference.generate import load_pipeline

    load_pipeline("model_id", "/lora/path")

    mock_pipe.vae.enable_tiling.assert_called_once()
    mock_pipe.enable_model_cpu_offload.assert_called_once()


# ── generate_video ───────────────────────────────────────────────────


@patch("src.inference.generate.export_to_video")
@patch("src.inference.generate.torch")
def test_generate_video_calls_pipeline(mock_torch, mock_export):
    """generate_video calls the pipeline with correct args."""
    mock_generator = MagicMock()
    mock_torch.Generator.return_value = mock_generator
    mock_generator.manual_seed.return_value = mock_generator

    mock_pipe = MagicMock()
    mock_output = MagicMock()
    mock_pipe.return_value = mock_output
    mock_output.frames = [[MagicMock()]]

    from src.inference.generate import generate_video

    result = generate_video(
        pipe=mock_pipe,
        prompt="ohwx person dancing",
        output_path="/out/video.mp4",
        height=480,
        width=832,
        num_frames=61,
        num_inference_steps=30,
        seed=42,
        fps=15,
    )

    mock_pipe.assert_called_once_with(
        prompt="ohwx person dancing",
        height=480,
        width=832,
        num_frames=61,
        num_inference_steps=30,
        generator=mock_generator,
    )
    mock_export.assert_called_once_with(
        mock_output.frames[0], "/out/video.mp4", fps=15
    )
    assert result == "/out/video.mp4"


@patch("src.inference.generate.export_to_video")
@patch("src.inference.generate.torch")
def test_generate_video_uses_seed(mock_torch, mock_export):
    """Generator is seeded with the provided seed."""
    mock_generator = MagicMock()
    mock_torch.Generator.return_value = mock_generator
    mock_generator.manual_seed.return_value = mock_generator

    mock_pipe = MagicMock()
    mock_pipe.return_value = MagicMock(frames=[[MagicMock()]])

    from src.inference.generate import generate_video

    generate_video(mock_pipe, "prompt", "/out.mp4", seed=999)

    mock_torch.Generator.assert_called_once_with("cpu")
    mock_generator.manual_seed.assert_called_once_with(999)
