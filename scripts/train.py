"""
Launch HunyuanVideo LoRA training.

Prerequisites:
  - Dataset prepared via `python -m scripts.prepare_dataset`
  - CUDA GPU available
  - HF_TOKEN set in .env for gated model access

Usage:
  python -m scripts.train
  python -m scripts.train --steps 2000 --rank 128
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    BATCH_SIZE,
    CAPTION_DROPOUT,
    CHECKPOINTING_STEPS,
    DATASET_DIR,
    FINETRAINERS_REPO,
    FINETRAINERS_TAG,
    GRAD_ACCUM_STEPS,
    HF_TOKEN,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    LR_SCHEDULER,
    MAX_GRAD_NORM,
    MODEL_ID,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SEED,
    TRAIN_STEPS,
    TRIGGER_TOKEN,
    WARMUP_STEPS,
)
from src.training.gpu_utils import get_supported_dtype, is_fp8_supported
from src.training.train import (
    detect_gpu,
    get_resolution_and_fp8,
    launch_training,
    setup_finetrainers,
)


def main():
    parser = argparse.ArgumentParser(description="Launch HunyuanVideo LoRA training")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Set HF token
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    else:
        print("WARNING: HF_TOKEN not set. You may not be able to download gated models.")

    # Detect GPU
    gpu_name, vram_gb = detect_gpu()
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram_gb:.1f} GB\n")

    if vram_gb == 0:
        print("ERROR: No CUDA GPU detected.")
        sys.exit(1)

    resolution_buckets, use_fp8 = get_resolution_and_fp8(
        vram_gb, fp8_supported=is_fp8_supported()
    )

    # Setup finetrainers
    finetrainers_dir = str(PROJECT_ROOT / "finetrainers")
    train_script = setup_finetrainers(
        install_dir=finetrainers_dir,
        repo_url=FINETRAINERS_REPO,
        tag=FINETRAINERS_TAG,
    )

    # Verify dataset
    videos_txt = DATASET_DIR / "videos.txt"
    prompts_txt = DATASET_DIR / "prompts.txt"
    if not videos_txt.exists() or not prompts_txt.exists():
        print("ERROR: Dataset not found. Run `python -m scripts.prepare_dataset` first.")
        sys.exit(1)

    with open(videos_txt) as f:
        n_videos = len(f.readlines())
    with open(prompts_txt) as f:
        n_prompts = len(f.readlines())

    print(f"\nDataset: {n_videos} videos, {n_prompts} captions")
    assert n_videos == n_prompts, f"Mismatch! {n_videos} videos vs {n_prompts} captions"
    print(f"Trigger: '{TRIGGER_TOKEN}'")
    print(f"LoRA rank: {args.rank}, LR: {args.lr}, Steps: {args.steps}")
    print(f"Resolution: {resolution_buckets}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Launch training
    mixed_precision = get_supported_dtype()
    launch_training(
        train_script_path=train_script,
        model_id=MODEL_ID,
        dataset_dir=str(DATASET_DIR),
        output_dir=str(OUTPUT_DIR),
        trigger_token=TRIGGER_TOKEN,
        resolution_buckets=resolution_buckets,
        use_fp8=use_fp8,
        mixed_precision=mixed_precision,
        train_steps=args.steps,
        lora_rank=args.rank,
        lora_alpha=args.rank,  # match rank
        learning_rate=args.lr,
        lr_scheduler=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        batch_size=args.batch_size,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        caption_dropout=CAPTION_DROPOUT,
        checkpointing_steps=CHECKPOINTING_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
