#!/usr/bin/env python3
"""Clean multi-GPU HunyuanVideo inference script.

Converted from `Notebooks/01. Test.ipynb` and cleaned for script usage.
"""

import base64
import os
import time

import torch
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video
from dotenv import load_dotenv
from huggingface_hub import login


def setup_auth() -> str:
    """Load environment variables and login to Hugging Face."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found in environment.")
    login(token=hf_token)
    return hf_token


def detect_gpus() -> tuple[int, float, torch.dtype]:
    """Detect available CUDA GPUs and select best dtype."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required - no GPU detected.")

    num_gpus = torch.cuda.device_count()
    total_vram = 0.0
    sm_major = torch.cuda.get_device_properties(0).major

    print(f"{'GPU':>5}  {'Name':<30}  {'VRAM':>8}  {'SM':>5}")
    print("-" * 56)
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1e9
        total_vram += vram_gb
        print(f"{i:>5}  {props.name:<30}  {vram_gb:>7.1f}G  {props.major}.{props.minor:>3}")

    dtype = torch.float16 if sm_major < 8 else torch.bfloat16
    torch.backends.cudnn.benchmark = True
    print(f"\n{'=' * 56}")
    print(f"GPUs: {num_gpus}  |  Total VRAM: {total_vram:.0f} GB  |  dtype: {dtype}")
    print(f"{'=' * 56}")
    return num_gpus, total_vram, dtype


def load_pipeline(hf_token: str, num_gpus: int, total_vram: float, dtype: torch.dtype) -> HunyuanVideoPipeline:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    per_gpu_mem = total_vram / num_gpus

    if num_gpus > 1:
        print(f"{num_gpus} GPUs ({total_vram:.0f} GB total) - distributing full precision model")

        # Leave ~4-5 GB per GPU free for activations during forward pass
        max_memory = {i: "11GiB" for i in range(num_gpus)}
        max_memory["cpu"] = "60GiB"  # allow overflow to CPU RAM

        from diffusers.quantizers import PipelineQuantizationConfig

        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": dtype,  # float16 for V100sdd
            },
            components_to_quantize=["transformer"],
        )

        # ~3-4 GB transformer + ~2 GB text encoders + VAE = plenty of room per GPU
        max_memory = {i: "14GiB" for i in range(num_gpus)}
        max_memory["cpu"] = "60GiB"

        pipe = HunyuanVideoPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map="balanced",
            token=hf_token,
        )

        pipe.enable_attention_slicing(slice_size="auto")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        return pipe

    if per_gpu_mem >= 45:
        print(f"{per_gpu_mem:.0f} GB VRAM - loading full precision model")
        pipe = HunyuanVideoPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            torch_dtype=torch.bfloat16,
            token=hf_token,
        )
        pipe.to("cuda")
        pipe.vae.enable_tiling()
        return pipe

    print(f"{per_gpu_mem:.0f} GB VRAM - loading quantized int4 model")
    from diffusers.quantizers import PipelineQuantizationConfig

    quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=["transformer"],
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def generate(pipe: HunyuanVideoPipeline) -> tuple[list, int, int, int]:
    """Generate video frames."""
    prompt = """
A Pomeranian sprints through a park between trees and benches, weaving playfully.
Dynamic tracking shot with smooth steadycam, quick but clean parallax, light motion blur,
sharp focus on the dog. Sun rays through leaves, cinematic contrast, realistic fur simulation
and paw impacts on grass. 4K, 60fps for smooth motion, 6 seconds. No text, no logos.
"""

    num_frames = 61
    height = 480
    width = 848
    steps = 30
    fps = 15

    torch.cuda.empty_cache()
    print(f"Generating {num_frames} frames at {width}x{height}...")
    start = time.time()
    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=steps,
    ).frames[0]
    elapsed = time.time() - start
    print(f"Done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    return output, width, height, fps


def save_video(output: list, width: int, height: int, fps: int) -> None:
    """Save generated video and optionally display inline in notebook contexts."""
    output_path = "hunyuan_output.mp4"
    export_to_video(output, output_path, fps=fps)
    print(f"Saved to {output_path}")

    try:
        from IPython.display import HTML, display

        with open(output_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        display(
            HTML(
                f"""
<video width="{width}" height="{height}" controls autoplay loop>
  <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
</video>
"""
            )
        )
    except Exception:
        # Running as a plain script (no notebook display).
        pass


def print_gpu_peaks() -> None:
    """Print peak memory per GPU and clear CUDA cache."""
    for i in range(torch.cuda.device_count()):
        peak = torch.cuda.max_memory_allocated(i) / 1e9
        print(f"GPU {i}: peak {peak:.1f} GB")
    torch.cuda.empty_cache()
    print("\nCUDA cache cleared")


def main() -> None:
    hf_token = setup_auth()
    num_gpus, total_vram, dtype = detect_gpus()
    pipe = load_pipeline(hf_token, num_gpus, total_vram, dtype)
    print("HunyuanVideo loaded.")
    output, width, height, fps = generate(pipe)
    save_video(output, width, height, fps)
    print_gpu_peaks()


if __name__ == "__main__":
    main()
