from datasets import load_dataset

from PIL import Image

import os

os.makedirs("data/train", exist_ok=True)

ds = load_dataset(
        "gvecchio/MatSynth",
        split = "train",
        streaming=True
        )

for idx, d in enumerate(ds):
    name = str(idx)
    path = "data/train"

    os.mkdir(path + "/" + name)

    basecolor = d['basecolor']
    height = d['height']
    normal = d['normal']
    roughness = d['roughness']

    basecolor_scaled = basecolor.resize((1024, 1024))
    height_scaled = height.resize((1024, 1024))
    normal_scaled = normal.resize((1024, 1024))
    roughness_scaled = roughness.resize((1024, 1024))

    basecolor_scaled.save(os.path.join(path, name, "basecolor.png"), "PNG")
    height_scaled.save(os.path.join(path, name, "height.png"), "PNG")
    normal_scaled.save(os.path.join(path, name, "normal.png"), "PNG")
    roughness_scaled.save(os.path.join(path, name, "roughness.png"), "PNG")

    with open(os.path.join(path, name, "desc.txt"), 'w') as file:
        file.write(f"Tags: {', '.join(d['metadata']['tags'])}\nName: {d['metadata']['name']}")

    print("Image No. {} -> {} saved".format(idx, d['metadata']['name']))

