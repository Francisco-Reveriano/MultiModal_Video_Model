# HunyuanVideo LoRA Fine-Tuning Pipeline

End-to-end pipeline for fine-tuning [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) with LoRA to generate personalized videos from text prompts. Uses a trigger-token approach (similar to DreamBooth) where a unique token (`ohwx`) is bound to a specific subject during training, then used at inference time to place that subject into novel scenes.

## Architecture

### Pipeline Phases

```mermaid
flowchart LR
    subgraph phase1 [Phase 1 -- Data Preparation]
        A[Analyze] --> B[Process]
        B --> D[Preview]
        B --> BA[Augment]
        BA --> D
        D --> E[Caption]
        E --> F[Validate]
    end

    subgraph phase2 [Phase 2 -- Training]
        G[finetrainers + accelerate]
        G --> H[LoRA Adapter]
    end

    subgraph phase3 [Phase 3 -- Inference]
        H --> PTV[Post-Training Validation]
        I[Base Model + LoRA] --> J[Generated Video]
    end

    F --> G
    H --> I
```

**Phase 1** transforms raw phone/camera footage into a training-ready dataset: normalized resolution, frame rate, duration, and text captions. An optional augmentation step (notebook 02A) creates temporal crops and horizontal flips to expand the dataset.

**Phase 2** fine-tunes the HunyuanVideo transformer with LoRA using the [finetrainers](https://github.com/huggingface/finetrainers) library, producing a lightweight adapter (~100 MB) instead of modifying the full 13B-parameter model. Metrics are logged to [Weights & Biases](https://wandb.ai). After training completes, a single validation video is generated automatically.

**Phase 3** loads the base HunyuanVideo model, applies the LoRA adapter, and generates new videos from text prompts containing the trigger token.

### Module Architecture

```mermaid
flowchart TD
    subgraph config_mod [config/]
        CFG[config.py]
    end

    subgraph src_mod [src/]
        VFT[video_fine_tuning.py]
        subgraph data_mod [data/]
            AN[analyze.py]
            PR[process.py]
            PV[preview.py]
            AUG[augment.py]
        end
        subgraph cap_mod [captioning/]
            GE[gemini.py]
            CA[augment.py]
        end
        subgraph train_mod [training/]
            TR[train.py]
            VA[validate.py]
            GP[gpu_utils.py]
        end
        subgraph infer_mod [inference/]
            GEN[generate.py]
        end
    end

    subgraph scripts_mod [scripts/]
        SPD[prepare_dataset.py]
        STR[train.py]
        SGN[generate.py]
        subgraph bash_mod [bash/]
            NOHUP[run_video_fine_tuning_nohup.sh]
        end
    end

    subgraph notebooks_mod [Notebooks/]
        NB1[01. Test.ipynb]
        NB2[02. Video Process Pipeline.ipynb]
        NB2A[02A. Augmentation Pipeline.ipynb]
        NB3[03. Video Fine-Tuning.ipynb]
        NB4[04. New Weights.ipynb]
    end

    CFG --> data_mod
    CFG --> cap_mod
    CFG --> train_mod
    CFG --> infer_mod
    SPD --> data_mod
    SPD --> cap_mod
    SPD --> train_mod
    STR --> train_mod
    SGN --> infer_mod
    NB2 --> data_mod
    NB2 --> cap_mod
    NB2A --> data_mod
    NB2A --> cap_mod
    NB3 --> train_mod
    NB4 --> infer_mod
    VFT --> train_mod
    NOHUP --> VFT
```

## Project Structure

```
MultiModal_Video_Model/
├── config/
│   ├── config.py              # All project settings (paths, hyperparams, API keys via .env)
│   └── config.py.example      # Sample config -- copy to config.py to get started
├── src/
│   ├── video_fine_tuning.py   # Standalone fine-tuning script (converted from notebook 03)
│   ├── data/
│   │   ├── analyze.py         # Video metadata extraction (resolution, fps, brightness)
│   │   ├── process.py         # Resize, crop, re-encode, trim via FFmpeg
│   │   ├── augment.py         # Temporal crop + horizontal flip augmentation
│   │   └── preview.py         # Thumbnail grid visualization
│   ├── captioning/
│   │   ├── gemini.py          # Auto-captioning with Gemini 2.5 Flash
│   │   └── augment.py         # Rule-based caption augmentation (flip rules, style variants)
│   ├── training/
│   │   ├── train.py           # finetrainers setup, training script generation, launch
│   │   ├── validate.py        # Dataset integrity checks before training
│   │   └── gpu_utils.py       # CUDA capability detection (bf16, fp8, compute capability)
│   └── inference/
│       └── generate.py        # Load HunyuanVideo + LoRA, generate video from prompt
├── scripts/
│   ├── prepare_dataset.py     # CLI: full data preparation pipeline
│   ├── train.py               # CLI: launch LoRA training
│   ├── generate.py            # CLI: generate video with trained adapter
│   └── bash/
│       ├── run_video_fine_tuning_nohup.sh  # Background launcher via nohup
│       └── README.md          # Launcher usage docs
├── Notebooks/
│   ├── 01. Test.ipynb                            # Quick model loading / generation test
│   ├── 02. Video Process Pipeline.ipynb          # Data preparation (no augmentation)
│   ├── 02A. Video Process Augmentation Pipeline.ipynb  # Data preparation with augmentation
│   ├── 03. Video Fine-Tuning.ipynb               # Fine-tuning with W&B logging
│   └── 04. New Weights.ipynb                     # Inference / validation with trained weights
├── finetrainers/              # Cloned finetrainers repo (not committed)
├── data/
│   ├── raw/                   # Input: raw video files (not committed)
│   ├── processed/             # Output: processed videos, captions.json, videos.txt, prompts.txt
│   ├── processed_aug/         # Output: augmented videos + captions (not committed)
│   └── Midpoint_Results/      # Post-training validation videos (not committed)
├── output/
│   ├── lora_weights/          # Trained LoRA checkpoints (not committed)
│   └── nohup_logs/            # Background launcher logs and PID files (not committed)
├── tests/                     # pytest suite mirroring src/ structure
├── requirements.txt
├── pyproject.toml
├── .env.example               # Template for required API keys
└── .gitignore
```

## Prerequisites

| Requirement | Details |
|---|---|
| Python | >= 3.10 |
| FFmpeg | Must be on `$PATH` (used for all video processing) |
| GPU | NVIDIA CUDA GPU (see VRAM tiers below) |
| Gemini API Key | For auto-captioning ([aistudio.google.com](https://aistudio.google.com)) |
| HuggingFace Token | For downloading the gated HunyuanVideo model |
| W&B API Key | For online training logging ([wandb.ai](https://wandb.ai)) |

### GPU VRAM Tiers

| VRAM | FP8 | Resolution Buckets | Max Frames | Notes |
|---|---|---|---|---|
| 80 GB+ | Off | 17x512x768, 49x512x768 | 49 | Full precision |
| 40 GB | Auto (sm_89+) | 17x512x768, 49x512x768 | 49 | FP8 upcasting on Ada/Hopper GPUs |
| 24 GB | Auto (sm_89+) | 17x512x768 | 17 | FP8 + reduced frame count |

FP8 requires compute capability >= 8.9 (RTX 4090, L40, H100). On older GPUs FP8 is disabled automatically.

## Setup

```bash
git clone <repo-url> && cd MultiModal_Video_Model

python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

# Secrets -- copy templates and fill in your keys
cp .env.example .env
cp config/config.py.example config/config.py
# Edit .env and set GEMINI_API_KEY, HF_TOKEN, and WANDB_API_KEY
```

Secrets (`GEMINI_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`) are loaded from the `.env` file via `python-dotenv`. Neither `.env` nor `config/config.py` are committed to the repo -- only the `.example` templates are tracked.

Place raw video files (`.mp4`, `.mov`, `.avi`) into `data/raw/`.

## Usage

### Option 1: Notebooks (recommended for exploration)

| Notebook | Purpose |
|---|---|
| `01. Test.ipynb` | Quick model loading and generation test |
| `02. Video Process Pipeline.ipynb` | Data preparation without augmentation |
| `02A. Video Process Augmentation Pipeline.ipynb` | Data preparation with temporal crops + flips |
| `03. Video Fine-Tuning.ipynb` | LoRA fine-tuning with W&B logging |
| `04. New Weights.ipynb` | Inference / validation with trained LoRA weights |

Run cells sequentially. Each pipeline step is a separate section with summary statistics.

For best training results, use notebook **02A** (with augmentation) to expand a small dataset before training. This multiplies each base video into up to 6 variants via temporal crops and horizontal flips.

### Option 2: CLI Scripts

```bash
# Step 1: Prepare dataset (analyze, process, caption, validate)
python -m scripts.prepare_dataset

# Step 2: Train LoRA adapter
python -m scripts.train

# Step 3: Generate video
python -m scripts.generate --prompt "ohwx running through a park, cinematic lighting"
```

Script flags:

```bash
python -m scripts.prepare_dataset --skip-caption    # Reuse existing captions
python -m scripts.prepare_dataset --skip-preview     # Headless / no display

python -m scripts.train --steps 2000 --rank 128 --lr 1e-4 --batch-size 1

python -m scripts.generate --lora-strength 0.8 --steps 50 --seed 123
```

> **Note:** The CLI `prepare_dataset.py` does not include augmentation. Use notebook **02A** for augmented dataset preparation.

### Option 3: Python API

```python
from src.data.analyze import analyze_all
from src.data.process import process_all
from src.data.augment import augment_processed_videos
from src.captioning.gemini import caption_all
from src.captioning.augment import caption_augmented_videos
from src.training.validate import validate_dataset
from src.inference.generate import load_pipeline, generate_video
```

### Option 4: Background Training (nohup)

Launch `src/video_fine_tuning.py` in the background with all output logged to a timestamped file:

```bash
bash scripts/bash/run_video_fine_tuning_nohup.sh
```

The launcher prints the PID, PID file, and log file path. Monitor and stop with:

```bash
# Follow logs in real time
tail -f output/nohup_logs/video_fine_tuning_<timestamp>.txt

# Stop the run
kill "$(cat output/nohup_logs/video_fine_tuning_<timestamp>.pid)"
```

## Pipeline Details

### 1. Analyze (`src/data/analyze.py`)

Scans every video in `data/raw/` and extracts metadata with OpenCV: resolution, frame rate, frame count, duration, mean brightness, and flags for black or overexposed frames. Videos shorter than `MIN_DURATION_SEC` are flagged.

### 2. Process (`src/data/process.py`)

Normalizes each video to training specs via FFmpeg:

- **Resize** to 768x512 (scale up to cover, then center-crop to exact dimensions)
- **Re-encode** at 24 fps, H.264, CRF 18, `yuv420p`
- **Trim** to the first 5 seconds (`MAX_DURATION_SEC`)
- **Strip audio** (`-an`)

Each output is validated against target specs before being accepted.

### 3. Preview (`src/data/preview.py`)

Extracts 4 evenly-spaced thumbnail frames per video and displays a matplotlib grid for visual QA.

### 4. Caption (`src/captioning/gemini.py`)

Extracts the mid-frame from each video, sends it to Gemini 2.5 Flash with a structured prompt that enforces the trigger token (`ohwx`) as subject identifier. Outputs:

- `captions.json` -- filename-to-caption mapping
- `videos.txt` -- one video path per line
- `prompts.txt` -- one caption per line (same order)

### 4A. Augment Videos (`src/data/augment.py`)

Creates augmented video variants from the processed base videos:

- **Temporal crops** -- 2 clips per video, 3 seconds each, evenly spaced across the original duration
- **Horizontal flip** -- mirrors the video left-to-right
- **Combined** -- flipped versions of each temporal crop

Per base video, this produces up to **6 variants**: 1 base + 2 temporal crops + 1 flipped base + 2 flipped crops. Output goes to `data/processed_aug/videos/`.

### 4B. Augment Captions (`src/captioning/augment.py`)

Applies rule-based transformations on top of the Gemini-generated base captions:

- **Flip rules** -- swaps "left" / "right" in captions for horizontally-flipped videos
- **Temporal crop descriptor** -- appends `"Short action-focused crop from the original sequence."`
- **Style variants** -- cycles through 3 cinematic style suffixes, selected deterministically by filename hash

Outputs: `captions_augmented.json`, `videos.txt`, `prompts.txt` in `data/processed_aug/`.

### 5. Validate (`src/training/validate.py`)

Pre-training integrity checks: resolution matches target, frame count >= 17, captions are non-empty and non-failed, video files are non-trivially sized, and `videos.txt` / `prompts.txt` line counts match.

### 6. Train (`src/training/train.py`, `src/video_fine_tuning.py`)

Clones [finetrainers](https://github.com/huggingface/finetrainers) v0.0.1, generates a training shell script, and launches it via `accelerate`. Key settings:

- LoRA on transformer attention layers (`to_q`, `to_k`, `to_v`, `to_out.0`)
- Rank 128, alpha 64
- AdamW optimizer, LR 5e-5 with cosine annealing schedule
- 50-step linear warmup
- bf16 mixed precision, gradient checkpointing, TF32 enabled
- 280 epochs (~630 optimizer steps with 9 videos)
- Batch size 1, gradient accumulation 4 (effective batch size 4)
- 10% caption dropout for classifier-free guidance
- Gradient clipping at 0.5
- Checkpoints saved every 200 steps (max 3 kept)
- Online logging to [Weights & Biases](https://wandb.ai) (`--report_to wandb`)

After training completes, a **post-training validation** step automatically loads the base model, applies the trained LoRA adapter (strength 0.6), and generates a validation video saved to `data/Midpoint_Results/post_training_validation.mp4`.

### 7. Inference (`src/inference/generate.py`, `Notebooks/04. New Weights.ipynb`)

Loads the base HunyuanVideo model, applies the trained LoRA adapter at a configurable strength (default 0.6), enables VAE tiling, then generates video from a text prompt. Notebook 04 provides an interactive workflow with automatic GPU detection, quantization selection, and LoRA weight discovery.

## Configuration

All settings live in `config/config.py` and read secrets from `.env` via `python-dotenv`.

### Video Processing

| Parameter | Default | Description |
|---|---|---|
| `TARGET_WIDTH` | 768 | Output video width |
| `TARGET_HEIGHT` | 512 | Output video height |
| `TARGET_FPS` | 24 | Output frame rate |
| `MAX_DURATION_SEC` | 5 | Max clip length |
| `MIN_FRAMES` | 17 | Minimum frame count for a valid clip |
| `TRIGGER_TOKEN` | `ohwx` | Subject identifier token |

### Augmentation

| Parameter | Default | Description |
|---|---|---|
| `TEMPORAL_CROP_DURATION_SEC` | 3.0 | Duration of each temporal crop (seconds) |
| `TEMPORAL_CROPS_PER_VIDEO` | 2 | Number of temporal crops per base video |
| `ENABLE_HORIZONTAL_FLIP` | True | Apply horizontal flip augmentation |
| `CAPTION_AUG_RULE_VARIANTS` | 2 | Number of caption style variants to cycle |

### Training

| Parameter | Default | Description |
|---|---|---|
| `TRAIN_EPOCHS` | 280 | Number of training epochs |
| `LORA_RANK` | 128 | LoRA rank (decomposition dimension) |
| `LORA_ALPHA` | 64 | LoRA scaling factor |
| `LEARNING_RATE` | 5e-5 | Peak learning rate after warmup |
| `LR_SCHEDULER` | `cosine` | Learning rate schedule |
| `WARMUP_STEPS` | 50 | Linear warmup steps |
| `BATCH_SIZE` | 1 | Micro-batch size per GPU |
| `GRAD_ACCUM_STEPS` | 4 | Gradient accumulation steps |
| `CAPTION_DROPOUT` | 0.1 | Caption drop probability (classifier-free guidance) |
| `MAX_GRAD_NORM` | 0.5 | Gradient clipping threshold |
| `CHECKPOINTING_STEPS` | 200 | Save checkpoint every N optimizer steps |
| `SEED` | 42 | Random seed |

### Inference

| Parameter | Default | Description |
|---|---|---|
| `LORA_STRENGTH` | 0.6 | Adapter weight at inference |
| `INFERENCE_HEIGHT` | 480 | Generated video height |
| `INFERENCE_WIDTH` | 832 | Generated video width |
| `INFERENCE_NUM_FRAMES` | 61 | Number of generated frames |
| `INFERENCE_STEPS` | 30 | Diffusion inference steps |

## Testing

Tests mirror the `src/` module structure under `tests/` and run with pytest:

```bash
pytest
```

Configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```
