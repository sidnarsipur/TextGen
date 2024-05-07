### Compare errors between ground truth normal and roughness maps and inference maps
import numpy as np
from PIL import Image

def cosine_error(gt, inf):
    gt = gt.flatten()
    inf = inf.flatten()
    return 1 - (gt @ inf) / (np.linalg.norm(gt) * np.linalg.norm(inf))

def rms_error(gt, inf):
    return np.sqrt(np.mean((gt - inf) ** 2))


if __name__ == "__main__":
    data_dir = "data/combined"
    # Cosine error for normal and roughness maps
    # RMS error for normal and roughness maps
    matforge_normal_cosine_errors = []
    matforge_roughness_cosine_errors = []
    controlnet_normal_cosine_errors = []
    controlnet_roughness_cosine_errors = []
    matforge_normal_rms_errors = []
    matforge_roughness_rms_errors = []
    controlnet_normal_rms_errors = []
    controlnet_roughness_rms_errors = []
    
    
    for i in range(89):
        
        # When comparing, make all images 512x512
        gt_normal = np.array(Image.open(f"{data_dir}/{i}/gt-normal.png").convert("RGB").resize((512, 512))).astype(np.float32) / 255
        gt_roughness = np.array(Image.open(f"{data_dir}/{i}/gt-roughness.png").convert("RGB").resize((512, 512))).astype(np.float32) / 255
        matforge_normal = np.array(Image.open(f"{data_dir}/{i}/matforge-normal.png").convert("RGB").resize((512, 512))).astype(np.float32) / 255
        matforge_roughness = np.array(Image.open(f"{data_dir}/{i}/matforge-roughness.png").convert("RGB").resize((512, 512))).astype(np.float32) / 255
        controlnet_normal = np.array(Image.open(f"{data_dir}/{i}/controlnet-normal.png").convert("RGB").resize((512, 512))).astype(np.float32) / 255
        controlnet_roughness = np.array(Image.open(f"{data_dir}/{i}/controlnet-roughness.png").convert("RGB").resize((512, 512))).astype(np.float32) / 255

        # Calculate cosine error for ground truth and inference normal maps
        matforge_normal_cosine_errors.append(cosine_error(gt_normal, matforge_normal))
        controlnet_normal_cosine_errors.append(cosine_error(gt_normal, controlnet_normal))
        # Calculate cosine error for ground truth and inference roughness maps
        matforge_roughness_cosine_errors.append(cosine_error(gt_roughness, matforge_roughness))
        controlnet_roughness_cosine_errors.append(cosine_error(gt_roughness, controlnet_roughness))
        # Calculate RMS error for ground truth and inference normal maps
        matforge_normal_rms_errors.append(rms_error(gt_normal, matforge_normal))
        controlnet_normal_rms_errors.append(rms_error(gt_normal, controlnet_normal))
        # Calculate RMS error for ground truth and inference roughness maps
        matforge_roughness_rms_errors.append(rms_error(gt_roughness, matforge_roughness))
        controlnet_roughness_rms_errors.append(rms_error(gt_roughness, controlnet_roughness))
       
    
    # Save the errors to a file
    with open("data/errors.txt", "w") as f:
        f.write(f"Average cosine error for matforge normal maps: {np.mean(matforge_normal_cosine_errors)}\n")
        f.write(f"Average cosine error for controlnet normal maps: {np.mean(controlnet_normal_cosine_errors)}\n")
        f.write(f"Average cosine error for matforge roughness maps: {np.mean(matforge_roughness_cosine_errors)}\n")
        f.write(f"Average cosine error for controlnet roughness maps: {np.mean(controlnet_roughness_cosine_errors)}\n")
        f.write(f"Average RMS error for matforge normal maps: {np.mean(matforge_normal_rms_errors)}\n")
        f.write(f"Average RMS error for controlnet normal maps: {np.mean(controlnet_normal_rms_errors)}\n")
        f.write(f"Average RMS error for matforge roughness maps: {np.mean(matforge_roughness_rms_errors)}\n")
        f.write(f"Average RMS error for controlnet roughness maps: {np.mean(controlnet_roughness_rms_errors)}\n")
    