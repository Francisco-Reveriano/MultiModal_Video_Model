import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

from src.training.gpu_utils import get_supported_torch_dtype


def load_pipeline(
    model_id: str,
    lora_path: str,
    lora_strength: float = 0.6,
) -> HunyuanVideoPipeline:
    """Load the HunyuanVideo pipeline with LoRA weights."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. HunyuanVideo requires a CUDA GPU for inference."
        )

    dtype = get_supported_torch_dtype()
    print(f"Loading base model (dtype={dtype})...")
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        model_id, transformer=transformer, torch_dtype=torch.float16
    )

    print(f"Loading LoRA from {lora_path} (strength={lora_strength})...")
    pipe.load_lora_weights(lora_path, adapter_name="hunyuan-lora")
    pipe.set_adapters(["hunyuan-lora"], [lora_strength])
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    return pipe


def generate_video(
    pipe: HunyuanVideoPipeline,
    prompt: str,
    output_path: str,
    height: int = 480,
    width: int = 832,
    num_frames: int = 61,
    num_inference_steps: int = 30,
    seed: int = 42,
    fps: int = 15,
) -> str:
    """Generate a video from a text prompt using the trained LoRA."""
    print(f"Generating: '{prompt}'")
    generator = torch.Generator("cpu").manual_seed(seed)

    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).frames[0]

    export_to_video(output, output_path, fps=fps)
    print(f"\nVideo saved to: {output_path}")
    return output_path
