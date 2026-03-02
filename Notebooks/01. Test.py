#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()


# In[ ]:


HF_TOKEN =  os.getenv("HF_TOKEN") # Replace with your token from https://huggingface.co/settings/tokens
login(token=HF_TOKEN)


# In[4]:


import torch
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video

assert torch.cuda.is_available(), "CUDA is required — no GPU detected!"

num_gpus = torch.cuda.device_count()
total_vram = 0
sm_major = torch.cuda.get_device_properties(0).major

print(f"{'GPU':>5}  {'Name':<30}  {'VRAM':>8}  {'SM':>5}")
print("-" * 56)
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_memory / 1e9
    total_vram += vram_gb
    print(f"{i:>5}  {props.name:<30}  {vram_gb:>7.1f}G  {props.major}.{props.minor:>3}")

per_gpu_mem = total_vram / num_gpus
dtype = torch.float16 if sm_major < 8 else torch.bfloat16

torch.backends.cudnn.benchmark = True

print(f"\n{'='*56}")
print(f"GPUs: {num_gpus}  |  Total VRAM: {total_vram:.0f} GB  |  dtype: {dtype}")
print(f"{'='*56}")


# In[ ]:


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if num_gpus > 1:
    # ===== MULTI-GPU — distribute model evenly, leaving headroom for activations =====
    print(f"🚀 {num_gpus} GPUs ({total_vram:.0f} GB total) — distributing full precision model")

    # Cap each GPU so weights spread across all GPUs instead of piling onto GPU 0.
    # The full model is ~43 GB in fp16 (LLaMA 16 GB + transformer 26 GB + VAE/CLIP ~1 GB).
    # 14 GiB × 8 = 112 GiB budget keeps everything on-GPU with ~2 GiB/GPU for activations.
    max_memory = {i: "14GiB" for i in range(num_gpus)}

    pipe = HunyuanVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        torch_dtype=dtype,
        device_map="balanced",
        max_memory=max_memory,
        token=HF_TOKEN,
    )
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

elif per_gpu_mem >= 45:
    # ===== A100 80GB — full precision on single GPU =====
    print(f"🚀 {per_gpu_mem:.0f} GB VRAM — loading full precision model")

    pipe = HunyuanVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    pipe.to("cuda")
    pipe.vae.enable_tiling()

else:
    # ===== Single GPU < 45GB — quantized (int4) =====
    print(f"💾 {per_gpu_mem:.0f} GB VRAM — loading quantized (int4) model (~14 GB)")

    from diffusers.quantizers import PipelineQuantizationConfig

    pipeline_quant_config = PipelineQuantizationConfig(
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
        quantization_config=pipeline_quant_config,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

print("✅ HunyuanVideo loaded!")


# In[ ]:


import time

# ✏️ EDIT YOUR PROMPT HERE
prompt = """
A Pomeranian sprints through a park between trees and benches, weaving playfully. Dynamic tracking shot with smooth steadycam, quick but clean parallax, light motion blur, sharp focus on the dog. Sun rays through leaves, cinematic contrast, realistic fur simulation and paw impacts on grass. 4K, 60fps for smooth motion, 6 seconds. No text, no logos.
"""

# ✏️ SETTINGS
NUM_FRAMES = 33        # 4k+1 rule: 33 frames / 15fps ≈ 2 sec
HEIGHT = 320           # Conservative first — bump once balance is confirmed
WIDTH = 576
STEPS = 30             # 30 is the recommended default
FPS = 15

# Clear any leftover memory before generation
torch.cuda.empty_cache()

print(f"🎬 Generating {NUM_FRAMES} frames at {WIDTH}x{HEIGHT}...")
start = time.time()

output = pipe(
    prompt=prompt,
    num_frames=NUM_FRAMES,
    height=HEIGHT,
    width=WIDTH,
    num_inference_steps=STEPS,
).frames[0]

elapsed = time.time() - start
print(f"✅ Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")


# In[ ]:


import base64
from IPython.display import HTML, display

output_path = "hunyuan_output.mp4"
export_to_video(output, output_path, fps=FPS)
print(f"✅ Saved to {output_path}")

with open(output_path, "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode()

display(HTML(f"""
<video width="{WIDTH}" height="{HEIGHT}" controls autoplay loop>
  <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
</video>
"""))


# In[ ]:


for i in range(torch.cuda.device_count()):
    peak = torch.cuda.max_memory_allocated(i) / 1e9
    print(f"GPU {i}: peak {peak:.1f} GB")

torch.cuda.empty_cache()
print("\n🧹 CUDA cache cleared")

