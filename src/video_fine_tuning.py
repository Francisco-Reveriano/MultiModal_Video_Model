# Auto-generated from Notebooks/03. Video Fine-Tuning.ipynb
# Converted by script; edit notebook/source of truth as needed.

# ===== Code Cell 0 =====
# ── Standard library & path setup ──────────────────────────────────────────────
import os
import sys
from pathlib import Path

# Third-party utilities for env-var management and HuggingFace authentication
from dotenv import load_dotenv
from huggingface_hub import login

# ── Project path ──────────────────────────────────────────────────────────────────
# This script lives in <root>/src/, so parent.parent is the project root.
# Inserting it into sys.path lets us import custom `config` and `src` packages.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Load .env file (e.g. HF_TOKEN, WANDB_API_KEY) from the project root.
load_dotenv()
# ── Import training constants from the central config ─────────────────────────────
# All tuneable defaults (model ID, paths, resolution, LoRA, trigger token, etc.)
# are defined once in config/config.py so every notebook stays in sync.
from config.config import (
    CAPTION_DROPOUT,      # probability of dropping a caption during training (classifier-free guidance)
    DATASET_DIR,          # directory containing videos.txt + prompts.txt
    FINETRAINERS_REPO,    # GitHub URL for the finetrainers training framework
    FINETRAINERS_TAG,     # pinned git tag to ensure reproducible builds
    INFERENCE_HEIGHT,     # video height for post-training validation
    INFERENCE_NUM_FRAMES, # number of frames for post-training validation
    INFERENCE_STEPS,      # diffusion steps for post-training validation
    INFERENCE_WIDTH,      # video width for post-training validation
    LORA_STRENGTH,        # LoRA adapter scaling for inference
    LR_SCHEDULER,         # learning-rate schedule type (e.g. "cosine")
    MAX_GRAD_NORM,        # gradient clipping threshold
    MIN_FRAMES,           # minimum number of frames a video must have
    MODEL_ID,             # HuggingFace model identifier for HunyuanVideo
    OUTPUT_DIR,           # where checkpoints and LoRA weights are saved
    SEED,                 # random seed for reproducibility
    TARGET_HEIGHT,        # target video height in pixels
    TARGET_WIDTH,         # target video width in pixels
    TRIGGER_TOKEN,        # special token associated with the fine-tuned concept
)

# ── HuggingFace authentication ────────────────────────────────────────────────────
# Required for downloading gated models (e.g. HunyuanVideo). The token is set in
# both env-var styles that different HF libraries expect, then used to log in.
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    login(token=HF_TOKEN)
else:
    print("WARNING: HF_TOKEN not set — gated model downloads may fail")

# ── Weights & Biases configuration ────────────────────────────────────────────────
# Hardcode project/entity here for reproducible run routing in your workspace.
WANDB_PROJECT = "multimodal-video-model"
WANDB_ENTITY = "francisco-reveriano-1-mckinsey-company"
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")

if not WANDB_API_KEY:
    raise ValueError("WANDB_API_KEY is required in .env for wandb.ai logging.")

os.environ["WANDB_API_KEY"] = WANDB_API_KEY
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_ENTITY"] = WANDB_ENTITY
os.environ["WANDB_MODE"] = "online"

# ── Summary printout ──────────────────────────────────────────────────────────────
print("Configuration")
print("=" * 50)
print(f"Model        : {MODEL_ID}")
print(f"Dataset      : {DATASET_DIR}")
print(f"Output       : {OUTPUT_DIR}")
print(f"Trigger token: {TRIGGER_TOKEN}")
print(f"W&B project  : {WANDB_PROJECT}")
print(f"W&B entity   : {WANDB_ENTITY}")

# ===== Code Cell 1 =====
import torch

# ── Guard: training requires at least one CUDA GPU ────────────────────────────────
assert torch.cuda.is_available(), "CUDA is required — no GPU detected!"

# Custom helpers that inspect GPU compute capability to pick precision & resolution
from src.training.gpu_utils import get_supported_dtype, is_fp8_supported
from src.training.train import get_resolution_and_fp8

# ── Enumerate all GPUs and collect hardware info ──────────────────────────────────
num_gpus = torch.cuda.device_count()
total_vram = 0
sm_major = torch.cuda.get_device_properties(0).major  # SM arch of primary GPU

# Print a formatted table of GPU index, name, VRAM, and SM version
print(f"{'GPU':>5}  {'Name':<30}  {'VRAM':>8}  {'SM':>5}")
print("-" * 56)
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_memory / 1e9
    total_vram += vram_gb
    print(f"{i:>5}  {props.name:<30}  {vram_gb:>7.1f}G  {props.major}.{props.minor:>3}")

# ── Determine training strategy based on hardware ─────────────────────────────────
per_gpu_mem = total_vram / num_gpus

# Pick the best precision the GPU supports (BF16 preferred, else FP16)
mixed_precision = get_supported_dtype()

# DeepSpeed ZeRO-3 sharding is needed when the full model doesn't fit on one GPU
# (heuristic: <35 GB per GPU for HunyuanVideo)
needs_sharding = num_gpus > 1 and per_gpu_mem < 35

# Choose resolution buckets (lower for low-VRAM GPUs) and whether to enable FP8
# quantized storage for the transformer weights to save memory
resolution_buckets, use_fp8 = get_resolution_and_fp8(
    per_gpu_mem, fp8_supported=is_fp8_supported()
)

# ── Print final hardware / strategy summary ──────────────────────────────────────
print(f"\n{'='*56}")
print(f"GPUs: {num_gpus}  |  Per-GPU: {per_gpu_mem:.0f} GB  |  Total: {total_vram:.0f} GB")
print(f"Mixed precision: {mixed_precision}  |  FP8: {use_fp8}")
print(f"Resolution buckets: {resolution_buckets}")
print(f"Needs DeepSpeed sharding: {needs_sharding}")
print(f"{'='*56}")

# ===== Code Cell 2 =====
from src.training.train import setup_finetrainers

# ── Clone / reuse the finetrainers repository ────────────────────────────────────
# setup_finetrainers() will:
#   1. Clone FINETRAINERS_REPO at the pinned FINETRAINERS_TAG (if not already present)
#   2. Install its Python dependencies into the current environment
#   3. Return the path to the train.py script that Accelerate will launch
finetrainers_dir = str(PROJECT_ROOT / "finetrainers")

train_script = setup_finetrainers(
    install_dir=finetrainers_dir,
    repo_url=FINETRAINERS_REPO,
    tag=FINETRAINERS_TAG,
)

# Confirm the resolved path so it's easy to inspect or run manually
print(f"\n✅ train.py: {train_script}")

# ===== Code Cell 3 =====
from src.training.validate import validate_dataset

# ── Locate the manifest files produced by notebook 02 ────────────────────────────
# videos.txt  — one video filename per line
# prompts.txt — one caption per line (same order as videos.txt)
videos_txt = DATASET_DIR / "videos.txt"
prompts_txt = DATASET_DIR / "prompts.txt"

# Fail early if the preprocessing notebook hasn't been run yet
assert videos_txt.exists(), f"videos.txt not found at {videos_txt} — run notebook 02 first!"
assert prompts_txt.exists(), f"prompts.txt not found at {prompts_txt} — run notebook 02 first!"

# ── Read and display basic dataset statistics ─────────────────────────────────────
with open(videos_txt) as f:
    video_lines = f.read().strip().split("\n")
with open(prompts_txt) as f:
    prompt_lines = f.read().strip().split("\n")

print(f"Videos  : {len(video_lines)}")
print(f"Captions: {len(prompt_lines)}")

# Show the first few captions as a quick sanity check
print(f"\nSample captions:")
for i, cap in enumerate(prompt_lines[:3]):
    print(f"  [{i}] {cap[:120]}...")

# ── Deep validation ───────────────────────────────────────────────────────────────
# Checks: file existence, video readability, resolution match, minimum frame count,
# and 1:1 alignment between videos and captions.
print()
is_valid = validate_dataset(
    str(DATASET_DIR), target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT, min_frames=MIN_FRAMES
)
assert is_valid, "Dataset validation failed — fix errors above before training!"

# ===== Code Cell 4 =====
# ✏️ TRAINING HYPERPARAMETERS — edit these before launching ─────────────────────
#
# These are the primary knobs that control training behaviour. Sensible defaults
# are provided, but you should adjust them based on your dataset size, GPU budget,
# and desired quality.

TRAIN_EPOCHS = 280           # 280 epochs, ~630 optimizer steps with grad_accum=4
LORA_RANK = 128               # Rank of the LoRA decomposition (higher = more capacity)
LORA_ALPHA = 64              # LoRA scaling factor (often set equal to rank)
LEARNING_RATE = 5e-5         # Peak learning rate after warmup
BATCH_SIZE = 1               # Micro-batch size per GPU (1 is typical for video)
GRAD_ACCUM_STEPS = 4         # Gradient accumulation steps (effective batch = 4 per optimizer step)
WARMUP_STEPS = 50            # ~5% of total steps for faster ramp-up
CHECKPOINTING_STEPS = 200    # Save a checkpoint every ~100 epochs (200 optimizer steps)

# ── Compute effective batch size for transparency ─────────────────────────────────
# effective_batch = num_gpus * micro_batch * gradient_accumulation
effective_batch = num_gpus * BATCH_SIZE * GRAD_ACCUM_STEPS

# ── Print summary so settings are logged alongside the notebook output ───────────
print("Training Settings")
print("=" * 50)
print(f"Epochs          : {TRAIN_EPOCHS}")
print(f"LoRA rank/alpha : {LORA_RANK}/{LORA_ALPHA}")
print(f"Learning rate   : {LEARNING_RATE}")
print(f"Batch size      : {BATCH_SIZE} per GPU x {num_gpus} GPUs x {GRAD_ACCUM_STEPS} accum = {effective_batch} effective")
print(f"Warmup steps    : {WARMUP_STEPS}")
print(f"Checkpoint every: {CHECKPOINTING_STEPS} steps")
print(f"LR scheduler    : {LR_SCHEDULER}")

# ===== Code Cell 5 =====
import json

# ── Create the output directory ───────────────────────────────────────────────────
output_dir = str(OUTPUT_DIR)
os.makedirs(output_dir, exist_ok=True)
midpoint_results_dir = PROJECT_ROOT / "data" / "Midpoint_Results"
midpoint_results_dir.mkdir(parents=True, exist_ok=True)

# ── Build Accelerate launch arguments ─────────────────────────────────────────────
# Three modes depending on hardware:
#   1. DeepSpeed ZeRO-3 — multi-GPU with model sharding (low per-GPU VRAM)
#   2. Simple DDP       — multi-GPU, model fits per GPU
#   3. Single GPU       — no distribution needed

if num_gpus > 1 and needs_sharding:
    # ── MODE 1: DeepSpeed ZeRO-3 ─────────────────────────────────────────────────
    # Shards parameters, gradients, and optimizer states across all GPUs.
    # CPU offloading is enabled for the optimizer to further reduce GPU memory.
    ds_config = {
        # Match precision to what was auto-detected earlier
        "bf16": {"enabled": mixed_precision == "bf16"},
        "fp16": {"enabled": mixed_precision == "fp16"},
        "zero_optimization": {
            "stage": 3,                                    # full param + grad + optimizer sharding
            "overlap_comm": True,                          # overlap communication with computation
            "contiguous_gradients": True,                  # reduce memory fragmentation
            "reduce_bucket_size": 5e7,                     # bytes per allreduce bucket
            "stage3_prefetch_bucket_size": 5e7,            # prefetch size for forward pass
            "stage3_param_persistence_threshold": 1e5,     # small params kept on all GPUs
            "stage3_gather_16bit_weights_on_model_save": True,  # consolidate for checkpoint save
            "offload_optimizer": {
                "device": "cpu",      # offload optimizer states to CPU RAM
                "pin_memory": True,   # pin CPU memory for faster GPU transfers
            },
        },
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "gradient_clipping": MAX_GRAD_NORM,
        "train_batch_size": BATCH_SIZE * num_gpus * GRAD_ACCUM_STEPS,  # global batch
        "train_micro_batch_size_per_gpu": BATCH_SIZE,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }

    # Write the DeepSpeed JSON config to the output directory
    ds_config_path = os.path.join(output_dir, "ds_config.json")
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f, indent=2)

    # Generate a matching Accelerate YAML config that references the DS config
    accel_config = (
        "compute_environment: LOCAL_MACHINE\n"
        "debug: false\n"
        "deepspeed_config:\n"
        f"  deepspeed_config_file: {ds_config_path}\n"
        "  zero3_init_flag: true\n"
        "distributed_type: DEEPSPEED\n"
        "downcast_bf16: 'no'\n"
        "machine_rank: 0\n"
        "main_training_function: main\n"
        f"mixed_precision: {mixed_precision}\n"
        "num_machines: 1\n"
        f"num_processes: {num_gpus}\n"
        "rdzv_backend: static\n"
        "same_network: true\n"
        "tpu_env: []\n"
        "tpu_use_cluster: false\n"
        "tpu_use_sudo: false\n"
        "use_cpu: false\n"
    )

    accel_config_path = os.path.join(output_dir, "accelerate_config.yaml")
    with open(accel_config_path, "w") as f:
        f.write(accel_config)

    accel_launch_args = f"--config_file {accel_config_path}"
    print(f"🚀 DeepSpeed ZeRO-3 config written:")
    print(f"   {ds_config_path}")
    print(f"   {accel_config_path}")

elif num_gpus > 1:
    # ── MODE 2: Multi-GPU DDP ────────────────────────────────────────────────────
    # Model fits on each GPU; use standard Distributed Data Parallel
    accel_launch_args = f"--multi_gpu --num_processes {num_gpus} --mixed_precision {mixed_precision}"
    print(f"🚀 Multi-GPU DDP on {num_gpus} GPUs")

else:
    # ── MODE 3: Single GPU ────────────────────────────────────────────────────────
    accel_launch_args = f"--mixed_precision {mixed_precision} --gpu_ids 0"
    print(f"🚀 Single GPU mode")

# Show the final launch prefix so it's easy to reproduce from the command line
print(f"\naccelerate launch {accel_launch_args}")

# ===== Code Cell 6 =====
import time
import subprocess

# ── Environment variables for training stability ──────────────────────────────────
BYPASS_HF_PROXY = 1
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduce CUDA OOM fragmentation
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_ENTITY"] = WANDB_ENTITY
os.environ["WANDB_MODE"] = "online"
os.environ["NCCL_P2P_DISABLE"] = "1"                                # avoid P2P issues on some GPUs
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"                    # suppress noisy NCCL logs
os.environ["FINETRAINERS_LOG_LEVEL"] = "DEBUG"                      # verbose finetrainers logging

# ── Optional FP8 flags ────────────────────────────────────────────────────────────
# When FP8 is enabled (SM ≥ 8.9 + enough VRAM), store transformer weights in
# float8_e4m3fn and skip certain modules that are sensitive to low precision.
fp8_flags = ""
if use_fp8:
    fp8_flags = (
        "    --layerwise_upcasting_modules transformer \\\n"
        "    --layerwise_upcasting_storage_dtype float8_e4m3fn \\\n"
        "    --layerwise_upcasting_skip_modules_pattern "
        'patch_embed pos_embed x_embedder context_embedder "^proj_in$" "^proj_out$" norm \\\n'
    )

# ── Assemble the full training shell script ──────────────────────────────────────
# This is written to disk so the exact command is reproducible outside the notebook.
venv_accelerate = os.path.join(PROJECT_ROOT, ".venv", "bin", "accelerate")

script = f"""#!/bin/bash
set -e

# Re-export env vars so the script is self-contained when run standalone
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="{WANDB_API_KEY}"
export WANDB_PROJECT="{WANDB_PROJECT}"
export WANDB_ENTITY="{WANDB_ENTITY}"
export WANDB_MODE=online
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=DEBUG

# Some environments inject local HTTP proxies that block Hugging Face downloads.
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
    {accel_launch_args} \\
    {train_script} \\
    --model_name hunyuan_video \\
    --pretrained_model_name_or_path {MODEL_ID} \\
    --data_root "{DATASET_DIR}" \\
    --video_column videos.txt \\
    --caption_column prompts.txt \\
    --id_token {TRIGGER_TOKEN} \\
    --video_resolution_buckets {resolution_buckets} \\
    --caption_dropout_p {CAPTION_DROPOUT} \\
    --dataloader_num_workers 2 \\
    --training_type lora \\
    --seed {SEED} \\
    --batch_size {BATCH_SIZE} \\
    --train_epochs {TRAIN_EPOCHS} \\
    --rank {LORA_RANK} \\
    --lora_alpha {LORA_ALPHA} \\
    --target_modules to_q to_k to_v to_out.0 \\
    --gradient_accumulation_steps {GRAD_ACCUM_STEPS} \\
    --gradient_checkpointing \\
    --max_grad_norm {MAX_GRAD_NORM} \\
    --optimizer adamw \\
    --lr {LEARNING_RATE} \\
    --lr_scheduler {LR_SCHEDULER} \\
    --lr_warmup_steps {WARMUP_STEPS} \\
    --enable_slicing \\
    --enable_tiling \\
    --precompute_conditions \\
    --allow_tf32 \\
    --checkpointing_steps {CHECKPOINTING_STEPS} \\
    --checkpointing_limit 3 \\
    --output_dir "{output_dir}" \\
    --report_to wandb \\
{fp8_flags}
"""

# Save the script and make it executable
script_path = os.path.join(output_dir, "run_training.sh")
with open(script_path, "w") as f:
    f.write(script)
os.chmod(script_path, 0o755)

# Path for the persistent log file (useful for post-mortem debugging)
log_path = os.path.join(output_dir, "training_log.txt")

print(f"Training script: {script_path}")
print(f"\n{'='*60}")
print("LAUNCHING TRAINING...")
print(f"{'='*60}\n")

# ── Execute training and stream output ────────────────────────────────────────────
# stdout and stderr are merged so all logs appear in order.
# Every line is printed to the notebook AND written to the log file.
start = time.time()

with open(log_path, "w") as log_file:
    proc = subprocess.Popen(
        ["bash", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        text=True,
    )
    # Stream line-by-line for real-time monitoring in the notebook
    for line in proc.stdout:
        print(line, end="")
        log_file.write(line)
    proc.wait()

elapsed = time.time() - start

# ── Report outcome ────────────────────────────────────────────────────────────────
if proc.returncode != 0:
    print(f"\n❌ Training failed (exit code {proc.returncode}) after {elapsed/60:.1f} min")
    print(f"Check log: {log_path}")
else:
    print(f"\n✅ Training complete in {elapsed/60:.1f} min")
    print(f"Log: {log_path}")

# ===== Code Cell 7: Post-training validation =====
# Generate a validation video using the trained LoRA weights.
# Training ran as a subprocess, so GPU memory is fully freed.
if proc.returncode == 0:
    from diffusers import HunyuanVideoPipeline
    from diffusers.utils import export_to_video

    print(f"\n{'='*60}")
    print("POST-TRAINING VALIDATION")
    print(f"{'='*60}\n")

    torch.cuda.empty_cache()
    validation_prompt = f"{TRIGGER_TOKEN} Pomeranian dog running in a sunny park, cinematic video"

    print(f"Loading pipeline from {MODEL_ID}...")
    pipe = HunyuanVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    pipe.to("cuda")
    pipe.vae.enable_tiling()

    print(f"Loading LoRA weights from {output_dir}...")
    pipe.load_lora_weights(output_dir)
    pipe.fuse_lora(lora_scale=LORA_STRENGTH)

    print(f"Generating {INFERENCE_NUM_FRAMES} frames at {INFERENCE_WIDTH}x{INFERENCE_HEIGHT}...")
    val_start = time.time()
    video_frames = pipe(
        prompt=validation_prompt,
        num_frames=INFERENCE_NUM_FRAMES,
        height=INFERENCE_HEIGHT,
        width=INFERENCE_WIDTH,
        num_inference_steps=INFERENCE_STEPS,
    ).frames[0]
    val_elapsed = time.time() - val_start

    val_video_path = midpoint_results_dir / "post_training_validation.mp4"
    export_to_video(video_frames, str(val_video_path), fps=15)
    print(f"\n✅ Validation video saved: {val_video_path} ({val_elapsed/60:.1f} min)")

    del pipe, video_frames
    torch.cuda.empty_cache()

# ===== Code Cell 8: Summary =====
output_path = Path(output_dir)

checkpoints = sorted(output_path.glob("checkpoint-*"))
if checkpoints:
    print(f"Checkpoints ({len(checkpoints)}):")
    for cp in checkpoints:
        size_mb = sum(f.stat().st_size for f in cp.rglob("*") if f.is_file()) / 1e6
        print(f"   {cp.name}  ({size_mb:.0f} MB)")
else:
    print("No checkpoints found")

lora_files = list(output_path.rglob("*.safetensors"))
if lora_files:
    print(f"\nLoRA weights:")
    for lf in lora_files:
        size_mb = lf.stat().st_size / 1e6
        print(f"   {lf.relative_to(output_path)}  ({size_mb:.1f} MB)")

if Path(log_path).exists():
    print(f"\nLast 10 lines of training log:")
    with open(log_path) as f:
        lines = f.readlines()
    for line in lines[-10:]:
        print(f"   {line.rstrip()}")

print(f"\nTotal training time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)")
print(f"Validation outputs: {midpoint_results_dir}")
