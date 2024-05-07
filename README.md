# Controlled Texture Map Generation

Texture maps have widespread use in video games, architecture and 3D modeling but are difficult to generate and design. 

We use [ControlNet](https://github.com/lllyasviel/ControlNet), a deep learning algorithm used for controlling image generation, and train it to generate normal and roughness texture maps, utilizing only the texture base color (Albedo Map) as the conditioning input.

# Dataset

We use [MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth), the largest open-source PBR materials dataset, created by Guiseppe Vecchio and Valentin Deschaire.

