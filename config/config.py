import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================
# Project root
# =============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================
# Data paths
# =============================================================
RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_DIR = PROJECT_ROOT / "data" / "processed"
VIDEO_DIR = DATASET_DIR / "videos"
DATASET_AUG_DIR = PROJECT_ROOT / "data" / "processed_aug"
VIDEO_AUG_DIR = DATASET_AUG_DIR / "videos"
OUTPUT_DIR = PROJECT_ROOT / "output" / "lora_weights"

# =============================================================
# Video processing specs
# =============================================================
TARGET_WIDTH = 768
TARGET_HEIGHT = 512
TARGET_FPS = 24
MAX_DURATION_SEC = 5
MIN_DURATION_SEC = 1
MIN_FRAMES = 17

# =============================================================
# Augmentation defaults
# =============================================================
TEMPORAL_CROP_DURATION_SEC = 3.0
TEMPORAL_CROPS_PER_VIDEO = 2
ENABLE_HORIZONTAL_FLIP = True
CAPTION_AUG_RULE_VARIANTS = 2

# =============================================================
# Captioning
# =============================================================
GEMINI_API_KEY = os.getenv("GOOGLE_API_STUDIO_KEY", "")
TRIGGER_TOKEN = "ohwx"
CAPTION_CONTEXT = """
Video includes a Pomeranian dog - 
"""
PER_VIDEO_CONTEXT = {}

# =============================================================
# Training
# =============================================================
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
FINETRAINERS_REPO = "https://github.com/huggingface/finetrainers.git"
FINETRAINERS_TAG = "v0.0.1"

TRAIN_STEPS = 1500
LORA_RANK = 128
LORA_ALPHA = 64
LEARNING_RATE = 5e-5
LR_SCHEDULER = "cosine"
WARMUP_STEPS = 30
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
CAPTION_DROPOUT = 0.1
CHECKPOINTING_STEPS = 200
MAX_GRAD_NORM = 0.5
SEED = 42

# =============================================================
# Inference
# =============================================================
INFERENCE_PROMPT = f"{TRIGGER_TOKEN} Pomeranian dog running in a sunny park, cinematic video"
LORA_STRENGTH = 0.6
INFERENCE_HEIGHT = 480
INFERENCE_WIDTH = 832
INFERENCE_NUM_FRAMES = 61
INFERENCE_STEPS = 30
