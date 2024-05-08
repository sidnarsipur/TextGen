# Controlled Texture Map Generation

Texture maps have widespread use in video games, architecture and 3D modeling but are difficult to generate and design. 

We use [ControlNet](https://github.com/lllyasviel/ControlNet), a deep learning algorithm used for controlling image generation, and train it to generate normal and roughness texture maps, utilizing only the base color (Albedo Map) as the conditioning input. Based on our tests, the models also works well on received material photographs.

# Dataset

We use [MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth), the largest open-source PBR materials dataset, created by Guiseppe Vecchio and Valentin Deschaire.

Our augmentated datatset is [available](https://huggingface.co/datasets/sidnarsipur/controlnet_data) on Hugging Face.

# Models

[controlnet-rough](https://huggingface.co/sidnarsipur/controlnet_rough)

[controlnet-normal](https://huggingface.co/sidnarsipur/controlnet_normal)

# Usage

See hugging face card for each model.

