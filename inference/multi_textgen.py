import argparse
import os

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

parser = argparse.ArgumentParser(description="Args for parser")
parser.add_argument("--seed", type=int, default=1, help="Seed for inference")
args = parser.parse_args()

base_model_path = "stabilityai/stable-diffusion-2-1-base"
controlnet_path = "sidnarsipur/controlnet_rough"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# control_image = load_image("inference/basecolor.png")
# prompt = "Normal Map"

# if control_image.size[0] > 2048 or control_image.size[1] > 2048:
#     control_image = control_image.resize((control_image.size[0] // 2, control_image.size[1] // 2))

# generator = torch.manual_seed(args.seed)

# image = pipe(
#     prompt, num_inference_steps=50, generator=generator, image=control_image
# ).images[0]
# image.save("inference/normal.png")

for i in range (89):
    control_image = load_image(f"data/test/{i}/basecolor.png")
    prompt = "Roughness Map"
    
    if control_image.size[0] > 2048 or control_image.size[1] > 2048:
        control_image = control_image.resize((512, 512))
    
    generator = torch.manual_seed(args.seed)
    
    image = pipe(
        prompt, num_inference_steps=25, generator=generator, image=control_image, height=512, width=512,
    ).images[0]
    os.makedirs(f"data/controlnet-inference/{i}", exist_ok=True)
    # if basecolor.png doesn't exist, create it
    if not os.path.exists(f"data/controlnet-inference/{i}/basecolor.png"):
        control_image.resize((512, 512)).save(f"data/controlnet-inference/{i}/basecolor.png")
    image.save(f"data/controlnet-inference/{i}/roughness.png")
    
    print(f"Saved roughness for {i}")
