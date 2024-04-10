#!/bin/bash
#SBATCH --job-name=Rough
#SBATCH --output=logs/output.txt
#SBATCH --error=logs/error.txt
#SBATCH -p gpu --gres=gpu -C A100

conda activate rough

accelerate launch diffusers/examples/controlnet/train_controlnet.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
--output_dir="models" \
--cache_dir="base_models" \
--dataset_name=sidnarsipur/roughness \
--conditioning_image_column=basecolor \
--image_column=roughness \
--caption_column=name \
--resolution=512 \
--learning_rate=1e-5 \
--proportion_empty_prompts=0.5 \
--validation_image "validation/basecolor_0.png" "validation/basecolor_557.png" "validation/basecolor_688.png" "validation/basecolor_688.png" \
--validation_prompt "Roughness Map of Rusty Grey Metal" "Roughness Map of Christmas Tree Ornament" "Roughness Map of Corrugated Steel" "Roughness Map of Fabric" \
--train_batch_size=4 \
--num_train_epochs=3 \
--tracker_project_name="controlnet" \
--enable_xformers_memory_efficient_attention \
--checkpointing_steps=5000 \
--validation_steps=5000 \
--logging_dir="logs" \
--report_to wandb \
--push_to_hub

conda deactivate rough

exit 0
