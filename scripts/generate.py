"""
Generate a video using trained HunyuanVideo LoRA weights.

Prerequisites:
  - Training completed via `python -m scripts.train`
  - CUDA GPU available

Usage:
  python -m scripts.generate
  python -m scripts.generate --prompt "ohwx dancing on a beach"
  python -m scripts.generate --lora-strength 0.8 --steps 50
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    INFERENCE_HEIGHT,
    INFERENCE_NUM_FRAMES,
    INFERENCE_PROMPT,
    INFERENCE_STEPS,
    INFERENCE_WIDTH,
    LORA_STRENGTH,
    MODEL_ID,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SEED,
)
from src.inference.generate import generate_video, load_pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate video with trained LoRA")
    parser.add_argument("--prompt", type=str, default=INFERENCE_PROMPT)
    parser.add_argument("--lora-strength", type=float, default=LORA_STRENGTH)
    parser.add_argument("--steps", type=int, default=INFERENCE_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. HunyuanVideo requires a CUDA GPU.")
        sys.exit(1)

    lora_path = str(OUTPUT_DIR)
    output_path = args.output or str(PROJECT_ROOT / "output" / "test_output.mp4")

    pipe = load_pipeline(
        model_id=MODEL_ID,
        lora_path=lora_path,
        lora_strength=args.lora_strength,
    )

    generate_video(
        pipe=pipe,
        prompt=args.prompt,
        output_path=output_path,
        height=INFERENCE_HEIGHT,
        width=INFERENCE_WIDTH,
        num_frames=INFERENCE_NUM_FRAMES,
        num_inference_steps=args.steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
