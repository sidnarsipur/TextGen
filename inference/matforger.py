import torch
import os
from PIL import Image

from diffusers import DiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = DiffusionPipeline.from_pretrained(
    "gvecchio/MatForger",
    trust_remote_code=True,
)

pipe.enable_vae_tiling()

pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)
pipe.to(device)

# model prompting with image
# Loop through data/test/{idx}/basecolor.png and create normal and roughness maps
for idx in range(89):
    prompt = Image.open(f"data/test/{idx}/basecolor.png")
    image = pipe(
        prompt,
        guidance_scale=6.0,
        height=512,
        width=512,
        tileable=True, # Allows to generate tileable materials
        # patched=False, # Reduce memory requirements for high-hes generation but affects quality 
        num_inference_steps=25,
    ).images[0]
    
    # get maps from prediction
    #basecolor = image.basecolor
    normal = image.normal
    roughness = image.roughness
    
    # save generated images
    # create data/matforge-inference/{idx} directory
    os.makedirs(f"data/matforge-inference/{idx}", exist_ok=True)
    # Save 512x512 basecolor image, original is 1024x1024
    prompt.resize((512, 512)).save(f"data/matforge-inference/{idx}/basecolor.png")
    normal.save(f"data/matforge-inference/{idx}/normal.png")
    roughness.save(f"data/matforge-inference/{idx}/roughness.png")
    print(f"Saved images for {idx}")

