import os
import subprocess
import tempfile
from typing import Optional

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
        install_req_file = req_file
        tmp_req_file = None
        with open(req_file, "r", encoding="utf-8") as f:
            req_lines = f.readlines()
        filtered_req_lines = [
            line
            for line in req_lines
            if not line.strip().lower().startswith("torchao")
        ]
        if len(filtered_req_lines) != len(req_lines):
            print("Skipping torchao install due to diffusers compatibility.")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp_file:
                tmp_file.writelines(filtered_req_lines)
                tmp_req_file = tmp_file.name
            install_req_file = tmp_req_file
        try:
            subprocess.run(
                ["pip", "install", "-q", "-r", install_req_file],
                check=True,
            )
        finally:
            if tmp_req_file and os.path.exists(tmp_req_file):
                os.remove(tmp_req_file)

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
    train_epochs: int = 1,
    train_steps: Optional[int] = None,
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
    validation_prompt: str = "",
    validation_steps: int = 100,
    num_validation_videos: int = 1,
    validation_frame_rate: int = 25,
    wandb_project: str = "",
    wandb_entity: str = "",
) -> str:
    """Build the training shell script. Returns the script content."""
    project_root = os.path.dirname(os.path.dirname(train_script_path))
    venv_accelerate = os.path.join(project_root, ".venv", "bin", "accelerate")
    fp8_flags = ""
    if use_fp8:
        fp8_flags = (
            "--layerwise_upcasting_modules transformer \\\n"
            "    --layerwise_upcasting_storage_dtype float8_e4m3fn \\\n"
            "    --layerwise_upcasting_skip_modules_pattern "
            'patch_embed pos_embed x_embedder context_embedder "^proj_in$" "^proj_out$" norm'
        )
    validation_flags = ""
    if validation_prompt:
        validation_flags = (
            "    --validation_prompts "
            f'"{validation_prompt}" \\\n'
            f"    --validation_steps {validation_steps} \\\n"
            f"    --num_validation_videos {num_validation_videos} \\\n"
            f"    --validation_frame_rate {validation_frame_rate} \\\n"
        )
    train_duration_flag = f"    --train_epochs {train_epochs} \\\n"
    if train_steps is not None:
        train_duration_flag = f"    --train_steps {train_steps} \\\n"

    script = f"""#!/bin/bash
set -e

export WANDB_MODE=online
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=DEBUG
export WANDB_PROJECT="{wandb_project}"
export WANDB_ENTITY="{wandb_entity}"

# Some environments inject local HTTP proxies that block Hugging Face model downloads.
# Set BYPASS_HF_PROXY=0 to keep existing proxy behavior.
if [ "${{BYPASS_HF_PROXY:-1}}" = "1" ]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    export NO_PROXY="${{NO_PROXY}},huggingface.co,cdn-lfs.huggingface.co,hf.co"
    export no_proxy="${{no_proxy}},huggingface.co,cdn-lfs.huggingface.co,hf.co"
fi

if [ -x "{venv_accelerate}" ]; then
    ACCELERATE_BIN="{venv_accelerate}"
else
    ACCELERATE_BIN="accelerate"
fi

"$ACCELERATE_BIN" launch \\
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
{train_duration_flag}
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
{validation_flags}
    --output_dir "{output_dir}" \\
    --report_to wandb \\
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
    wandb_api_key = os.getenv("WANDB_API_KEY", "")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY is required for online wandb logging.")

    os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["WANDB_MODE"] = "online"
    wandb_project = kwargs.get("wandb_project", "")
    wandb_entity = kwargs.get("wandb_entity", "")
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity
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
