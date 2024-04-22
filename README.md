# Controlled Texture Map Generation

Texture maps have widespread use in video games, architecture and 3D modeling but are difficult to generate and design. 

We use [ControlNet](https://github.com/lllyasviel/ControlNet), a deep learning algorithm used for controlling image generation, and train it to generate mettalic, normal and roughness texture maps, utilizing only the texture base color (Albedo Map) as the conditioning input. This project involved making a copy of the Stable Diffusion model and fine-tuning it on our curated dataset.

# Dataset

We use [MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth), the largest open-source PBR materials dataset, created by Guiseppe Vecchio and Valentin Deschaire.

From the ~6k images we receive from MatSynth, we perform rotation, flip and crop to create a final dataset with ~96,000 pairs.


