import os
import subprocess
import tempfile

import torch


def detect_gpu() -> tuple[str, float]:
    """Detect GPU name and VRAM in GB."""
    if not torch.cuda.is_available():
        return "CPU", 0.0
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return name, vram_gb


def get_resolution_and_fp8(
    vram_gb: float, fp8_supported: bool = True
) -> tuple[str, bool]:
    """Choose resolution buckets and FP8 based on available VRAM and GPU capability.

    Args:
        vram_gb: Available VRAM in GB.
        fp8_supported: Whether the GPU supports FP8 (sm_89+). When False,
            FP8 is never enabled regardless of VRAM.
    """
    if vram_gb >= 60:
        print(f"80GB+ VRAM -- full precision, up to 49 frames")
        return "17x512x768 49x512x768", False
    elif vram_gb >= 35:
        use_fp8 = fp8_supported
        label = "FP8 upcasting" if use_fp8 else "no FP8 (unsupported)"
        print(f"40GB VRAM -- {label}, up to 49 frames")
        return "17x512x768 49x512x768", use_fp8
    else:
        use_fp8 = fp8_supported
        label = "FP8" if use_fp8 else "no FP8 (unsupported)"
        print(f"Low VRAM ({vram_gb:.0f}GB) -- {label} + 17 frames only")
        return "17x512x768", use_fp8


def setup_finetrainers(
    install_dir: str,
    repo_url: str,
    tag: str,
) -> str:
    """Clone finetrainers repo and install its requirements. Returns train.py path."""
    if os.path.exists(os.path.join(install_dir, "train.py")):
        print(f"Finetrainers already installed at {install_dir}")
        return os.path.join(install_dir, "train.py")

    print(f"Cloning finetrainers {tag}...")
    subprocess.run(
        ["git", "clone", "--branch", tag, "--depth", "1", repo_url, install_dir],
        check=True,
    )

    req_file = os.path.join(install_dir, "requirements.txt")
    if os.path.exists(req_file):
        print("Installing finetrainers requirements...")
        subprocess.run(
            ["pip", "install", "-q", "-r", req_file],
            check=True,
        )

    train_script = os.path.join(install_dir, "train.py")
    assert os.path.exists(train_script), f"train.py not found at {train_script}"
    print(f"train.py ready at {train_script}")
    return train_script


def build_training_script(
    train_script_path: str,
    model_id: str,
    dataset_dir: str,
    output_dir: str,
    trigger_token: str,
    resolution_buckets: str,
    use_fp8: bool,
    mixed_precision: str = "bf16",
    train_steps: int = 1500,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    learning_rate: float = 2e-4,
    lr_scheduler: str = "constant_with_warmup",
    warmup_steps: int = 100,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    caption_dropout: float = 0.05,
    checkpointing_steps: int = 500,
    max_grad_norm: float = 1.0,
    seed: int = 42,
) -> str:
    """Build the training shell script. Returns the script content."""
    fp8_flags = ""
    if use_fp8:
        fp8_flags = (
            "--layerwise_upcasting_modules transformer \\\n"
            "    --layerwise_upcasting_storage_dtype float8_e4m3fn \\\n"
            "    --layerwise_upcasting_skip_modules_pattern "
            'patch_embed pos_embed x_embedder context_embedder "^proj_in$" "^proj_out$" norm'
        )

    script = f"""#!/bin/bash
set -e

export WANDB_MODE=offline
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=DEBUG

accelerate launch \\
    --mixed_precision={mixed_precision} \\
    --gpu_ids=0 \\
    {train_script_path} \\
    --model_name hunyuan_video \\
    --pretrained_model_name_or_path {model_id} \\
    --data_root "{dataset_dir}" \\
    --video_column videos.txt \\
    --caption_column prompts.txt \\
    --id_token {trigger_token} \\
    --video_resolution_buckets {resolution_buckets} \\
    --caption_dropout_p {caption_dropout} \\
    --dataloader_num_workers 2 \\
    --training_type lora \\
    --seed {seed} \\
    --batch_size {batch_size} \\
    --train_steps {train_steps} \\
    --rank {lora_rank} \\
    --lora_alpha {lora_alpha} \\
    --target_modules to_q to_k to_v to_out.0 \\
    --gradient_accumulation_steps {grad_accum_steps} \\
    --gradient_checkpointing \\
    --max_grad_norm {max_grad_norm} \\
    --optimizer adamw \\
    --lr {learning_rate} \\
    --lr_scheduler {lr_scheduler} \\
    --lr_warmup_steps {warmup_steps} \\
    --enable_slicing \\
    --enable_tiling \\
    --precompute_conditions \\
    --allow_tf32 \\
    --checkpointing_steps {checkpointing_steps} \\
    --checkpointing_limit 3 \\
    --output_dir "{output_dir}" \\
    --report_to none \\
    {fp8_flags}
"""
    return script


def launch_training(
    train_script_path: str,
    model_id: str,
    dataset_dir: str,
    output_dir: str,
    trigger_token: str,
    resolution_buckets: str,
    use_fp8: bool,
    mixed_precision: str = "bf16",
    **kwargs,
) -> None:
    """Write and execute the training shell script."""
    os.makedirs(output_dir, exist_ok=True)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    os.environ["FINETRAINERS_LOG_LEVEL"] = "DEBUG"

    script = build_training_script(
        train_script_path=train_script_path,
        model_id=model_id,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        trigger_token=trigger_token,
        resolution_buckets=resolution_buckets,
        use_fp8=use_fp8,
        mixed_precision=mixed_precision,
        **kwargs,
    )

    script_path = os.path.join(output_dir, "run_training.sh")
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    print(f"Training script written to {script_path}")
    print(f"\n{'=' * 60}")
    print("LAUNCHING TRAINING...")
    print(f"{'=' * 60}\n")

    log_path = os.path.join(output_dir, "training_log.txt")
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            ["bash", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {proc.returncode}")

    print(f"\nTraining complete! Log saved to {log_path}")
