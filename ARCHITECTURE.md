# Teaching a Video Model to Know Your Dog: Architecture of a HunyuanVideo LoRA Fine-Tuning Pipeline

## The Problem

Text-to-video generation has reached a turning point. Models like [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) -- a 13-billion-parameter diffusion transformer from Tencent -- can produce photorealistic 4-second clips from nothing but a text prompt. But these models generate _generic_ subjects. Ask for "a dog running in a park" and you'll get a plausible dog, but never _your_ dog.

Fine-tuning a 13B-parameter model from scratch to learn one subject is computationally absurd. This project solves that problem with a production-quality pipeline that takes 9 raw phone clips of a Pomeranian, and through a sequence of automated steps -- video normalization, AI captioning, data augmentation, LoRA training, and post-training validation -- produces a lightweight adapter that teaches HunyuanVideo exactly what "ohwx" looks like.

The result: a 625 MB LoRA file that, when merged at 60% strength into the 13B base model, lets you write `"ohwx running through snow, cinematic lighting"` and get a video of _that specific Pomeranian_ running through snow.

---

## Architecture Overview

The pipeline operates in three phases, each designed to be independently executable, reproducible, and recoverable from failure.

```
Phase 1: Data Preparation          Phase 2: Training              Phase 3: Inference
┌─────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────┐
│  Raw clips (9 videos)   │   │  finetrainers v0.0.1     │   │  Base HunyuanVideo   │
│         │                │   │  + Accelerate launcher   │   │  + LoRA adapter       │
│    ┌────▼────┐           │   │         │                │   │         │              │
│    │ Analyze │ OpenCV    │   │    ┌────▼────┐           │   │    ┌────▼────┐        │
│    └────┬────┘           │   │    │ Validate│ Dataset   │   │    │ Generate│        │
│    ┌────▼────┐           │   │    └────┬────┘           │   │    └────┬────┘        │
│    │ Process │ FFmpeg    │   │    ┌────▼────┐           │   │    ┌────▼────┐        │
│    └────┬────┘           │   │    │ Precomp │ Cache     │   │    │ Export  │ MP4    │
│    ┌────▼────┐           │   │    └────┬────┘ embeds    │   │    └─────────┘        │
│    │ Caption │ Gemini    │   │    ┌────▼────┐           │   └──────────────────────┘
│    └────┬────┘           │   │    │  Train  │ 280 ep    │
│    ┌────▼────┐           │   │    └────┬────┘           │
│    │ Augment │ 6x expand │   │    ┌────▼────┐           │
│    └────┬────┘           │   │    │  Post-  │ Validate  │
│    ┌────▼────┐           │   │    │  Train  │           │
│    │Validate │ Integrity │   │    └─────────┘           │
│    └─────────┘           │   └──────────────────────────┘
└─────────────────────────┘
```

What makes this architecture distinctive isn't any single component -- it's how the components compose. The captioning system uses a mid-frame extraction strategy because Gemini works on images, not video. The augmentation pipeline uses deterministic filename hashing so results are reproducible without storing metadata. The training step runs as a subprocess so the parent process can reclaim GPU memory for validation. Each design choice cascades into the next.

---

## Component Deep Dive

### 1. Video Analysis (`src/data/analyze.py`)

The pipeline begins with forensics. Before touching a single pixel, every file in `data/raw/` is profiled using OpenCV:

- **Temporal metadata**: FPS, frame count, duration, aspect ratio
- **Photometric quality**: Mean and standard deviation of brightness at the mid-frame
- **Failure flags**: `is_likely_black` (mean brightness < 15), `is_likely_overexposed` (mean > 240)

This isn't just logging -- it's a gate. Videos that are too short (`< MIN_DURATION_SEC`), corrupt (zero FPS, unreadable), or visually degenerate get flagged before any compute-intensive processing begins. The analysis function returns structured dictionaries with `[OK]`, `[WARN]`, and `[FAIL]` tags, giving operators a clear go/no-go signal.

### 2. Video Normalization (`src/data/process.py`)

HunyuanVideo expects training data at precise specifications. The normalization pipeline uses FFmpeg via subprocess to enforce them:

```
ffmpeg -y -i input.mp4 -t 5 \
  -vf "fps=24,scale=768:512:force_original_aspect_ratio=increase,crop=768:512" \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an -movflags +faststart \
  output.mp4
```

The filter chain is carefully ordered:

1. **`fps=24`** -- Re-timestamps frames to exactly 24 fps. Phone footage varies wildly (24, 29.97, 30, 60 fps); this normalizes temporal density.
2. **`scale=768:512:force_original_aspect_ratio=increase`** -- Scales up to _cover_ the target dimensions without distortion. A 1920x1080 source becomes 768x432 (too short vertically), so it scales to 910x512 instead.
3. **`crop=768:512`** -- Center-crops the overflow. No letterboxing, no stretching -- just a clean center extraction.
4. **`-an`** -- Strips audio entirely. Video diffusion models don't use audio, and it wastes bytes.
5. **`-crf 18`** -- Near-lossless quality. The model will eventually see this through a VAE encoder; introducing compression artifacts here would compound with quantization noise downstream.

After encoding, a second OpenCV pass validates the output: correct resolution, FPS within tolerance (allowing float rounding), minimum 17 frames, no all-black frames. Files that fail validation are deleted automatically.

### 3. AI Captioning (`src/captioning/gemini.py`)

This is where the pipeline gets interesting. HunyuanVideo's training loop expects paired `(video, text_caption)` data. Writing captions by hand for even 9 videos is tedious and inconsistent. Instead, the pipeline uses **Google Gemini 2.5 Flash** to generate structured captions from a single representative frame.

The strategy: extract the _exact mid-frame_ from each processed video, encode it as JPEG bytes, and send it to Gemini with a prompt that enforces:

- The trigger token `ohwx` as the subject identifier (always the opening word)
- Coverage of: appearance, action/pose, setting, lighting, camera angle, implied motion
- 2-3 sentences, specific and visual
- No real names, no subjective opinions

A real caption from the training dataset:

> ohwx, a golden-furred Pomeranian dog, is captured mid-motion on a light-colored tiled floor. The dog appears to be rolling or shaking its body vigorously, causing significant motion blur across its head and tail. Bright, even lighting illuminates the scene, which is viewed from a slightly elevated camera angle looking down at ohwx.

The mid-frame strategy is a deliberate tradeoff. A single frame captures the visual essence of a short clip (3-5 seconds) well enough for training captions, while avoiding the complexity and cost of video-native captioning APIs. The `CAPTION_CONTEXT` config parameter (e.g., `"Video includes a Pomeranian dog"`) provides Gemini with domain knowledge so it doesn't waste tokens identifying the species.

Three files are written atomically: `captions.json` (full mapping with failure records), `videos.txt` (paths), and `prompts.txt` (captions) -- the last two in lockstep order, which is exactly what finetrainers expects.

### 4. Data Augmentation

With only 9 source videos, overfitting is inevitable without augmentation. The pipeline includes a two-stage augmentation system that multiplies the dataset by up to 6x.

#### 4A. Video Augmentation (`src/data/augment.py`)

For each base video, the pipeline generates:

| Variant | Method | Purpose |
|---|---|---|
| Base | Identity (no copy) | Original processed clip |
| Temporal crop 1 | FFmpeg `-ss {t1} -t 3` | 3-second window from early in the clip |
| Temporal crop 2 | FFmpeg `-ss {t2} -t 3` | 3-second window from later in the clip |
| Horizontal flip | FFmpeg `-vf hflip` | Mirror image of the full clip |
| Flip + crop 1 | Both transforms | Mirror of the first temporal window |
| Flip + crop 2 | Both transforms | Mirror of the second temporal window |

The temporal crop start times are computed with an even-spacing formula:

```python
start_time = round((i / (num_crops - 1)) * max_start, 3)
```

This distributes crops across the available duration rather than randomly sampling, which guarantees coverage of the full clip without overlap when `num_crops` is small.

#### 4B. Caption Augmentation (`src/captioning/augment.py`)

Raw caption duplication would teach the model that all variants are identical. Instead, captions are modified to reflect each transformation:

1. **Flip rules**: Regex-based `\bleft\b` <-> `\bright\b` swaps (case-insensitive) for mirrored videos. If the original says "the dog faces left," the flipped variant says "the dog faces right."

2. **Temporal crop descriptor**: Appends `"Short action-focused crop from the original sequence."` -- this teaches the model that shorter clips represent focused action segments rather than full scenes.

3. **Style variants**: Appends one of three cinematic style suffixes:
   - `"Cinematic composition, natural lighting, realistic motion detail."`
   - `"Photorealistic detail, clean motion continuity, dynamic camera movement."`
   - `"High-fidelity visual detail, stable framing, and smooth subject motion."`

The style variant is selected **deterministically** using `hashlib.md5(filename).hexdigest()` modulo the variant count. This is an underappreciated design choice: it makes the augmentation fully reproducible without storing any per-file state. The same filename always maps to the same style variant across runs, across machines, across Python versions.

### 5. The Trigger Token: Why "ohwx"

The token `ohwx` is central to the entire approach. Borrowed from the DreamBooth technique, it exploits a property of large language models: rare tokens have weak prior associations.

`ohwx` is a 4-character string that almost certainly never appeared in HunyuanVideo's pretraining corpus. By prepending it to every training caption (`"ohwx, a golden-furred Pomeranian dog..."`), the model learns to associate `ohwx` exclusively with the visual appearance of the training subject. At inference time, including `ohwx` in the prompt activates this learned association.

The key insight is that this works _because_ the token is meaningless. A common word like "Pomeranian" already has strong associations from pretraining data -- fine-tuning on it would fight against those priors. An invented token like `ohwx` starts as a blank slate, making it far easier for LoRA's low-rank updates to bind it to a specific visual concept.

### 6. HunyuanVideo's Dual Text Encoder

HunyuanVideo uses an unusual two-encoder architecture that directly influences how training and inference work:

**Primary encoder: LlamaModel** -- A full large language model serves as the main text understanding backbone. Prompts are wrapped in a chat template (`<|start_header_id|>system...<|eot_id|><|start_header_id|>user...<|eot_id|>`), and hidden states from the penultimate layer are extracted with the first 95 tokens (system prompt) cropped off. This gives the transformer rich, contextual text embeddings with no practical token limit.

**Secondary encoder: CLIPTextModel** -- A standard CLIP text encoder with a hard 77-token limit, providing pooled text projections. This is the component that truncates long prompts -- a constraint that directly influenced the pipeline's prompt design (keeping inference prompts under 77 tokens).

Both embeddings, plus a guidance scalar (multiplied by 1000), are passed to the diffusion transformer at every denoising step. During training with `--precompute_conditions`, both encoders run _once_ at the start, caching their outputs as `.pt` files. This eliminates the encoders from the training loop entirely -- they can even be unloaded from GPU after precomputation, freeing VRAM for the diffusion process.

### 7. LoRA Training Architecture

The training system is built on three layers of abstraction:

```
Python orchestrator (video_fine_tuning.py)
    └── Generated bash script (run_training.sh)
         └── accelerate launch
              └── finetrainers train.py
                   └── PEFT LoRA on HunyuanVideoTransformer3DModel
```

#### Why LoRA and Not Full Fine-Tuning

HunyuanVideo's transformer has 13 billion parameters. Full fine-tuning would require:
- ~52 GB just for fp32 optimizer states (Adam stores 2 copies of each parameter)
- ~26 GB for bf16 model weights
- Gradients, activations, and intermediate states on top

LoRA (Low-Rank Adaptation) decomposes each weight update as `W' = W + BA` where `B` is `d x r` and `A` is `r x d`, with `r = 128` much smaller than `d` (the hidden dimension, typically 3072 for HunyuanVideo). Only `B` and `A` are trained -- the original 13B weights are frozen.

The target modules are the four attention projection matrices in every transformer block: `to_q`, `to_k`, `to_v`, `to_out.0`. These are the matrices that determine _what the model attends to_ -- modifying them teaches the model to "look for" and "reconstruct" the features that define `ohwx`.

With rank 128 and alpha 64, the effective scaling is `alpha / rank = 0.5`, meaning the LoRA update is applied at half magnitude. This acts as implicit regularization, preventing the adapter from overwhelming the base model's learned priors.

#### The Subprocess Architecture

A critical and unusual design decision: training is launched as a **child subprocess** via `subprocess.Popen(["bash", script_path])`. The parent Python process streams stdout line-by-line to both the terminal and a log file.

This is not just a convenience -- it solves a real GPU memory problem:

1. **Memory isolation**: When the subprocess exits, the OS reclaims _all_ of its GPU memory. The parent process can then load the full HunyuanVideo pipeline for validation without competing for VRAM.

2. **Crash recovery**: The generated bash script at `output/lora_weights/run_training.sh` is fully self-contained (all environment variables baked in). If the notebook kernel dies, the training script can be re-launched independently.

3. **Log persistence**: The training log at `output/lora_weights/training_log.txt` survives kernel crashes, SSH disconnects, and notebook restarts.

4. **Reproducibility**: The bash script is a complete record of the exact training command, including all hyperparameters, paths, and environment variables. It can be copied to a different machine and re-run as-is.

#### Training Hyperparameters

The current configuration reflects iterative tuning over multiple training runs:

| Parameter | Value | Rationale |
|---|---|---|
| Epochs | 280 | ~630 optimizer steps with 9 videos and grad_accum=4 |
| Rank | 128 | Higher rank captures more subject detail (tradeoff: 625 MB adapter) |
| Alpha | 64 | 0.5x scaling for regularization |
| Learning rate | 5e-5 | Conservative peak; cosine decay to ~0 |
| Warmup | 50 steps | ~8% of total steps, prevents early instability |
| Gradient accumulation | 4 | Effective batch size 4 from micro-batch 1 |
| Caption dropout | 10% | Randomly drops captions for classifier-free guidance |
| Gradient clipping | 0.5 | Tighter than default (1.0) to prevent spikes |
| Checkpoints | Every 200 steps | ~100 epochs between saves, max 3 retained |

The **cosine annealing** schedule was chosen over constant or linear decay because it spends more time at low learning rates near convergence, allowing the model to settle into a better minimum. The training log confirms this: loss dropped from ~0.08 to ~0.036, with gradient norms shrinking to 0.000733 by the final step -- indicating the model reached a stable optimum.

#### FP8 Layerwise Upcasting

On GPUs with compute capability >= 8.9 (Ada Lovelace, Hopper), the pipeline enables FP8 layerwise upcasting. This stores transformer weights as `float8_e4m3fn` (8-bit) but casts them to `bfloat16` before each forward pass, then back to FP8 for storage.

Critically, certain modules are _excluded_ from FP8:

- `patch_embed`, `pos_embed` -- Positional encoding layers where precision loss causes spatial artifacts
- `x_embedder`, `context_embedder` -- Input projection layers at the boundary between encoders and transformer
- `proj_in`, `proj_out` -- Gate layers at module boundaries
- All `norm` layers -- Normalization statistics are highly sensitive to quantization

This surgical exclusion pattern cuts transformer VRAM by roughly half while preserving training stability. On the H100 used for this project (85 GB), FP8 wasn't needed -- but the automatic detection ensures the pipeline works on 40 GB GPUs (A100-40G, RTX 4090) without code changes.

#### Precomputed Conditions

The `--precompute_conditions` flag is one of the most impactful optimizations. On the first epoch, finetrainers runs both text encoders and the VAE encoder on every training sample, saving the results as `.pt` files:

```
data/processed/hunyuan_video_..._precomputed/
├── conditions/
│   ├── video_001_processed-0.pt    # Text embeddings
│   ├── video_002_processed-0.pt
│   └── ...
└── latents/
    ├── video_001_processed-0.pt    # VAE-encoded video latents
    ├── video_002_processed-0.pt
    └── ...
```

For all subsequent epochs (279 of them), training loads directly from cache. This eliminates the two text encoders and the VAE encoder from the training loop, freeing their combined ~45 GB of VRAM for the diffusion process. With 9 videos and 280 epochs, this saves 279 x 9 = 2,511 redundant forward passes through three large models.

### 8. Post-Training Validation

After training completes, the parent process (which has been streaming logs) immediately runs validation:

```python
pipe = HunyuanVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.load_lora_weights(output_dir)
pipe.fuse_lora(lora_scale=0.6)

video = pipe(
    prompt="ohwx Pomeranian dog running in a sunny park, cinematic video",
    num_frames=61, height=480, width=832, num_inference_steps=30
).frames[0]

export_to_video(video, "data/Midpoint_Results/post_training_validation.mp4", fps=15)
```

The `fuse_lora()` call permanently merges the adapter into the base weights at 60% strength. This is intentional: at full strength (1.0), LoRA adapters trained on small datasets tend to over-saturate the base model's style, producing artifacts. At 0.6, the subject identity is strong but the base model's motion priors and scene composition remain intact.

This validation step is only possible because of the subprocess architecture. The training process exited and released its GPU memory, giving the parent process a clean 85 GB slate to load the full inference pipeline.

### 9. Multi-GPU Strategy

The training system adapts to available hardware automatically:

| Scenario | Strategy | Details |
|---|---|---|
| 1 GPU, >= 60 GB | Single GPU, bf16 | Full precision, no quantization |
| 1 GPU, 35-60 GB | Single GPU, FP8 | Layerwise upcasting on sm_89+ GPUs |
| 1 GPU, < 35 GB | Single GPU, FP8 | Reduced resolution (17 frames max) |
| N GPUs, >= 35 GB each | Multi-GPU DDP | Standard data-parallel training |
| N GPUs, < 35 GB each | DeepSpeed ZeRO-3 | Full parameter/gradient/optimizer sharding + CPU offload |

For inference, Notebook 04 implements a different multi-GPU strategy: the transformer's 60 attention blocks are split across GPUs using `device_map="auto"` with per-GPU memory limits, while the text encoders and VAE are dispatched to separate GPUs. This layer-parallel approach is more memory-efficient for inference than training-style DDP.

---

## What Makes This Unique

### 1. End-to-End Automation From Phone to Personalized Video

Most video fine-tuning guides stop at "prepare your dataset." This pipeline starts with raw phone footage and handles every step: analysis, normalization, captioning, augmentation, validation, training, and inference. The entire path from `data/raw/` to a generated video is executable with a single notebook run or a single background script.

### 2. 9-Video Training Through Intelligent Augmentation

Training a video diffusion model on 9 clips sounds impossible. The augmentation pipeline makes it work by expanding those 9 clips to up to 54 variants, each with semantically-adjusted captions. The temporal crops teach the model different action segments within each clip. The horizontal flips double the visual diversity. The caption variants prevent the model from memorizing exact text-to-visual mappings.

### 3. Subprocess Training With Memory Reclamation

Running training as a subprocess is unconventional in the ML world, where most frameworks expect in-process execution. But it solves a real problem: on a single GPU, you can't run training (which consumes ~38 GB) and inference (which needs ~45 GB) in the same process. The subprocess boundary provides a clean memory fence that no amount of `torch.cuda.empty_cache()` can replicate.

### 4. Deterministic Reproducibility Without State Files

The caption augmentation system achieves reproducibility through filename hashing rather than random seeds or state files. The same input files always produce the same augmented outputs, regardless of execution order, parallelism, or platform. This is a subtle but important property for debugging: if a training run produces unexpected results, you can trace any augmented caption back to its deterministic generation rule.

### 5. Pre-Mocked Test Suite

The test suite (`tests/`) runs without GPU, FFmpeg, API keys, or any ML dependencies. The `conftest.py` pre-registers mock objects into `sys.modules` before any source imports, creating a completely synthetic environment. This enables CI/CD pipelines on CPU-only runners while still exercising the full logic of every module.

### 6. Self-Documenting Training Commands

The generated `run_training.sh` script serves as both the execution mechanism and the experiment record. Every hyperparameter, environment variable, and path is baked into the script. This means experiment tracking doesn't depend on W&B being online or config files being preserved -- the bash script itself is a complete, re-executable record of the training configuration.

---

## Training Results

The pipeline was validated on a single NVIDIA H100 (85 GB) with 9 raw videos of a Pomeranian dog:

| Metric | Value |
|---|---|
| Base videos | 9 |
| Augmented variants | Up to 54 (6x expansion) |
| Training epochs | 280 |
| Optimizer steps | ~630 (840 forward passes / 4 grad accum) |
| Wall clock time | ~13.5 hours |
| Final loss | 0.0365 |
| Final gradient norm | 0.000733 |
| Peak GPU memory | 38.9 GB (of 85 GB available) |
| LoRA adapter size | 625 MB (rank 128) |
| Inference time | ~3.5 minutes per 61-frame video |

The cosine learning rate schedule fully decayed to near-zero by the final step, and the gradient norm dropped three orders of magnitude from early training, indicating stable convergence.

---

## Technical Stack

| Layer | Technology | Role |
|---|---|---|
| Video processing | FFmpeg, OpenCV | Normalization, metadata extraction, frame sampling |
| Captioning | Google Gemini 2.5 Flash | Automated text caption generation from mid-frames |
| Model | HunyuanVideo (13B params) | Base text-to-video diffusion transformer |
| Fine-tuning | PEFT LoRA, finetrainers v0.0.1 | Low-rank adapter training |
| Distributed training | Accelerate, DeepSpeed | Multi-GPU and memory optimization |
| Mixed precision | bf16, FP8 (optional) | VRAM reduction with minimal quality loss |
| Experiment tracking | Weights & Biases | Loss, gradient norms, learning rate curves |
| Quantization | bitsandbytes NF4 | 4-bit inference on lower-VRAM GPUs |
| Configuration | python-dotenv, config module | Centralized settings with secret isolation |
| Testing | pytest with pre-mocked conftest | GPU-free, dependency-free test execution |

---

## Conclusion

This pipeline demonstrates that personalized video generation is achievable with minimal data and a single GPU. The architectural choices -- subprocess training, deterministic augmentation, precomputed condition caching, trigger-token binding -- aren't individually revolutionary, but their composition creates a system that's greater than the sum of its parts. Nine phone clips of a dog become a model that can place that exact dog in any scene you can describe.

The LoRA adapter approach means this can be repeated for any subject -- a person, a product, a pet, a place -- without retraining the base model. Swap the videos, adjust the `CAPTION_CONTEXT`, and run the pipeline again. The architecture is the constant; the subject is the variable.
