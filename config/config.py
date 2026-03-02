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
# Captioning
# =============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TRIGGER_TOKEN = "ohwx"
CAPTION_CONTEXT = """
These videos are of a top colombian influencer and instagram model.
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
LORA_RANK = 64
LORA_ALPHA = 64
LEARNING_RATE = 2e-4
LR_SCHEDULER = "constant_with_warmup"
WARMUP_STEPS = 100
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
CAPTION_DROPOUT = 0.05
CHECKPOINTING_STEPS = 500
MAX_GRAD_NORM = 1.0
SEED = 42

# =============================================================
# Inference
# =============================================================
INFERENCE_PROMPT = (
    f"{TRIGGER_TOKEN} A sexy colombian latin girl dancing to spanish music "
    "in a thong bikini, realistic style, cinematic lighting, 4K quality."
)
LORA_STRENGTH = 0.6
INFERENCE_HEIGHT = 480
INFERENCE_WIDTH = 832
INFERENCE_NUM_FRAMES = 61
INFERENCE_STEPS = 30
