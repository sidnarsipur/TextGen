from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder

repo_id = "sidnarsipur/controlnet_normal"
base_model = "stabilityai/stable-diffusion-2-1-base"
repo_folder = "controlnet_normal"

upload_folder(
    repo_id=repo_id,
    folder_path=repo_folder,
    commit_message="Update New Model",
    ignore_patterns=["step_*", "epoch_*"],
)