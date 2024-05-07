                                                                    #!/bin/bash
#SBATCH --job-name=Normal
#SBATCH --error=logs/error.txt
#SBATCH --output=logs/error.txt
#SBATCH --time=0-8:00
#SBATCH -c 2 --mem=132G
#SBATCH -p gpu --gres=gpu:1 -C A100

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate rough

accelerate launch diffusers/examples/controlnet/train_controlnet.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
--controlnet_model_name_or_path="sidnarsipur/controlnet_normal" \
--output_dir="controlnet_normal" \
--cache_dir=base_models \
--dataset_name="base_models/sidnarsipur___controlnet_data" \
--conditioning_image_column=basecolor \
--image_column=normal \
--resolution=512 \
--learning_rate=1e-4 \
--proportion_empty_prompts=1.0 \
--empty_prompt_sub="Normal Map" \
--validation_image "validation/basecolor_88.png" "validation/basecolor_128.png" "validation/basecolor_176.png" "validation/basecolor_$
--validation_prompt "Normal Map" "Normal Map" "Normal Map" "Normal Map" "Normal Map" \
--train_batch_size=12 \
--num_train_epochs=3 \
--tracker_project_name="controlnet_normal" \
--enable_xformers_memory_efficient_attention \
--checkpointing_steps=2600 \
--report_to wandb \
--validation_steps=1300 \
--logging_dir="logs" \
--push_to_hub

conda deactivate
exit 0

