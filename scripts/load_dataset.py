import os

from datasets import DownloadConfig, load_dataset

# Check the last image that was saved
last_idx = 0
for idx in range(100000):
    if not os.path.exists(f"data/train/{idx}"):
        last_idx = idx - 1
        break
print(f"Last image saved: {last_idx}")


# Load HuggingFace Dataset
ds = load_dataset(
    "gvecchio/MatSynth",
    split="train",
    streaming=True,
    download_config=DownloadConfig(resume_download=True),
    download_mode="reuse_dataset_if_exists",
)

# Create train directory
os.makedirs("data/train", exist_ok=True)

# Loop over dataset
for idx, d in enumerate(ds):
    name = str(idx)
    path = "data/train"

    # Create data directory
    # skip if already exists, redo the last one that already exists
    if not os.path.exists(os.path.join(path, name)):
        os.mkdir(os.path.join(path, name))
    else:
        print("Image No. {} -> {} already exists".format(idx, d["metadata"]["name"]))
        continue

    # Load Maps
    basecolor = d["basecolor"]
    height = d["height"]
    normal = d["normal"]
    roughness = d["roughness"]

    # Resize to 1024x1024
    basecolor_scaled = basecolor.resize((1024, 1024))
    height_scaled = height.resize((1024, 1024))
    normal_scaled = normal.resize((1024, 1024))
    roughness_scaled = roughness.resize((1024, 1024))

    # Save resized maps locally as PNG
    basecolor_scaled.save(os.path.join(path, name, "basecolor.png"), "PNG")
    height_scaled.save(os.path.join(path, name, "height.png"), "PNG")
    normal_scaled.save(os.path.join(path, name, "normal.png"), "PNG")
    roughness_scaled.save(os.path.join(path, name, "roughness.png"), "PNG")

    # Save metadata locally
    with open(os.path.join(path, name, "desc.txt"), "w") as file:
        file.write(
            f"Tags: {', '.join(d['metadata']['tags'])}\nName: {d['metadata']['name']}"
        )

    # Confirmation Print
    print("Image No. {} -> {} saved".format(idx, d["metadata"]["name"]))
