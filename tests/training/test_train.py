"""Tests for src.training.train — GPU detection, training script generation."""

import os
from unittest.mock import MagicMock, patch

import pytest


# ── detect_gpu ───────────────────────────────────────────────────────


@patch("src.training.train.torch")
def test_detect_gpu_with_cuda(mock_torch):
    """Returns GPU name and VRAM when CUDA available."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
    props = MagicMock()
    props.total_memory = 80 * 1024**3  # 80 GB
    mock_torch.cuda.get_device_properties.return_value = props

    from src.training.train import detect_gpu

    name, vram = detect_gpu()
    assert name == "NVIDIA A100"
    assert abs(vram - 80.0) < 0.1


@patch("src.training.train.torch")
def test_detect_gpu_no_cuda(mock_torch):
    """Returns CPU, 0.0 when no CUDA."""
    mock_torch.cuda.is_available.return_value = False

    from src.training.train import detect_gpu

    name, vram = detect_gpu()
    assert name == "CPU"
    assert vram == 0.0


# ── get_resolution_and_fp8 ──────────────────────────────────────────


def test_resolution_high_vram():
    """80GB+ VRAM: full precision, no FP8."""
    from src.training.train import get_resolution_and_fp8

    res, fp8 = get_resolution_and_fp8(80.0, fp8_supported=True)
    assert "49x512x768" in res
    assert fp8 is False


def test_resolution_mid_vram_fp8_supported():
    """40GB VRAM with FP8 support: uses FP8."""
    from src.training.train import get_resolution_and_fp8

    res, fp8 = get_resolution_and_fp8(40.0, fp8_supported=True)
    assert "49x512x768" in res
    assert fp8 is True


def test_resolution_mid_vram_fp8_unsupported():
    """40GB VRAM without FP8: no FP8 despite VRAM."""
    from src.training.train import get_resolution_and_fp8

    res, fp8 = get_resolution_and_fp8(40.0, fp8_supported=False)
    assert "49x512x768" in res
    assert fp8 is False


def test_resolution_low_vram_fp8_supported():
    """Low VRAM with FP8 support: FP8, 17 frames only."""
    from src.training.train import get_resolution_and_fp8

    res, fp8 = get_resolution_and_fp8(16.0, fp8_supported=True)
    assert "17x512x768" in res
    assert "49x512x768" not in res
    assert fp8 is True


def test_resolution_low_vram_fp8_unsupported():
    """Low VRAM without FP8: no FP8, 17 frames only."""
    from src.training.train import get_resolution_and_fp8

    res, fp8 = get_resolution_and_fp8(16.0, fp8_supported=False)
    assert "17x512x768" in res
    assert fp8 is False


# ── build_training_script ────────────────────────────────────────────


def test_build_script_contains_key_flags():
    """Generated script includes all critical flags."""
    from src.training.train import build_training_script

    script = build_training_script(
        train_script_path="/path/train.py",
        model_id="model/HunyuanVideo",
        dataset_dir="/data",
        output_dir="/output",
        trigger_token="ohwx",
        resolution_buckets="17x512x768",
        use_fp8=False,
    )
    assert "--mixed_precision=bf16" in script
    assert "--model_name hunyuan_video" in script
    assert "--training_type lora" in script
    assert "--id_token ohwx" in script
    assert "--gradient_checkpointing" in script
    assert "accelerate launch" in script


def test_build_script_fp8_flags():
    """FP8 mode includes layerwise upcasting flags."""
    from src.training.train import build_training_script

    script = build_training_script(
        train_script_path="/path/train.py",
        model_id="model/HunyuanVideo",
        dataset_dir="/data",
        output_dir="/output",
        trigger_token="ohwx",
        resolution_buckets="17x512x768",
        use_fp8=True,
    )
    assert "--layerwise_upcasting_modules" in script
    assert "float8_e4m3fn" in script


def test_build_script_no_fp8_flags():
    """Non-FP8 mode excludes layerwise upcasting flags."""
    from src.training.train import build_training_script

    script = build_training_script(
        train_script_path="/path/train.py",
        model_id="model/HunyuanVideo",
        dataset_dir="/data",
        output_dir="/output",
        trigger_token="ohwx",
        resolution_buckets="17x512x768",
        use_fp8=False,
    )
    assert "--layerwise_upcasting_modules" not in script


def test_build_script_custom_params():
    """Custom parameters are reflected in script."""
    from src.training.train import build_training_script

    script = build_training_script(
        train_script_path="/path/train.py",
        model_id="model/HunyuanVideo",
        dataset_dir="/data",
        output_dir="/output",
        trigger_token="ohwx",
        resolution_buckets="17x512x768",
        use_fp8=False,
        train_steps=3000,
        lora_rank=128,
        learning_rate=1e-4,
        seed=123,
    )
    assert "--train_steps 3000" in script
    assert "--rank 128" in script
    assert "--lr 0.0001" in script
    assert "--seed 123" in script


def test_build_script_mixed_precision_fp16():
    """mixed_precision parameter is respected."""
    from src.training.train import build_training_script

    script = build_training_script(
        train_script_path="/path/train.py",
        model_id="model/HunyuanVideo",
        dataset_dir="/data",
        output_dir="/output",
        trigger_token="ohwx",
        resolution_buckets="17x512x768",
        use_fp8=False,
        mixed_precision="fp16",
    )
    assert "--mixed_precision=fp16" in script
    assert "--mixed_precision=bf16" not in script


# ── setup_finetrainers ──────────────────────────────────────────────


def test_setup_finetrainers_already_installed(tmp_path):
    """Skips clone when train.py already exists."""
    (tmp_path / "train.py").touch()

    from src.training.train import setup_finetrainers

    result = setup_finetrainers(str(tmp_path), "https://repo", "v0.0.1")
    assert result == str(tmp_path / "train.py")


@patch("src.training.train.subprocess")
def test_setup_finetrainers_clones_repo(mock_subprocess, tmp_path):
    """Clones repo and installs requirements."""
    install_dir = str(tmp_path / "finetrainers")

    # After clone, simulate train.py appearing
    def fake_run(cmd, check=False):
        if cmd[0] == "git":
            os.makedirs(install_dir, exist_ok=True)
            open(os.path.join(install_dir, "train.py"), "w").close()
            open(os.path.join(install_dir, "requirements.txt"), "w").close()

    mock_subprocess.run.side_effect = fake_run

    from src.training.train import setup_finetrainers

    result = setup_finetrainers(install_dir, "https://repo", "v0.0.1")
    assert result.endswith("train.py")
    assert mock_subprocess.run.call_count == 2  # git clone + pip install


# ── launch_training ──────────────────────────────────────────────────


@patch("src.training.train.subprocess")
def test_launch_training_success(mock_subprocess, tmp_path):
    """Successful training writes script and completes."""
    mock_proc = MagicMock()
    mock_proc.stdout = iter(["Training...\n", "Done\n"])
    mock_proc.returncode = 0
    mock_subprocess.Popen.return_value = mock_proc

    from src.training.train import launch_training

    launch_training(
        train_script_path="/path/train.py",
        model_id="model/HV",
        dataset_dir="/data",
        output_dir=str(tmp_path),
        trigger_token="ohwx",
        resolution_buckets="17x512x768",
        use_fp8=False,
    )

    assert os.path.exists(tmp_path / "run_training.sh")
    assert os.path.exists(tmp_path / "training_log.txt")


@patch("src.training.train.subprocess")
def test_launch_training_failure_raises(mock_subprocess, tmp_path):
    """Non-zero exit code raises RuntimeError."""
    mock_proc = MagicMock()
    mock_proc.stdout = iter(["Error\n"])
    mock_proc.returncode = 1
    mock_subprocess.Popen.return_value = mock_proc

    from src.training.train import launch_training

    with pytest.raises(RuntimeError, match="exit code 1"):
        launch_training(
            train_script_path="/path/train.py",
            model_id="model/HV",
            dataset_dir="/data",
            output_dir=str(tmp_path),
            trigger_token="ohwx",
            resolution_buckets="17x512x768",
            use_fp8=False,
        )


@patch("src.training.train.subprocess")
def test_launch_training_passes_mixed_precision(mock_subprocess, tmp_path):
    """mixed_precision parameter is passed through to script."""
    mock_proc = MagicMock()
    mock_proc.stdout = iter(["OK\n"])
    mock_proc.returncode = 0
    mock_subprocess.Popen.return_value = mock_proc

    from src.training.train import launch_training

    launch_training(
        train_script_path="/path/train.py",
        model_id="model/HV",
        dataset_dir="/data",
        output_dir=str(tmp_path),
        trigger_token="ohwx",
        resolution_buckets="17x512x768",
        use_fp8=False,
        mixed_precision="fp16",
    )

    script_content = (tmp_path / "run_training.sh").read_text()
    assert "--mixed_precision=fp16" in script_content
