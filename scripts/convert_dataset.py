from datasets import Dataset, load_dataset
from pathlib import Path
from PIL import Image

import json
import re
import random
from random import shuffle
from datasets import Dataset

def generate_name(entry_id):
    with open(f"data/transformed/{entry_id}/desc.json") as file:
        desc = json.load(file)
        
        name = desc["Name"]
        name = re.sub(r'^\s+|\d+|[A-Z](?!\w)', '', name)
        name = re.sub(r'\s+', ' ', name.lstrip())
        name = "Roughness Map of " + name

    return name

def entry_for_id(entry_id):
    roughness = Image.open(f"data/transformed/{entry_id}/roughness.png")
    basecolor = Image.open(f"data/transformed/{entry_id}/basecolor.png")
    name = generate_name(entry_id)

    return {
        "basecolor": basecolor,
        "roughness": roughness,
        "name": name
    }

def generate_entries():
    rng = list(range(0, 96899+1))
    shuffle(rng)

    for x in rng:
        yield entry_for_id(x)

ds = Dataset.from_generator(generate_entries)
ds.push_to_hub("sidnarsipur/roughness", token="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")

# ds = load_dataset("sidnarsipur/roughness", split="train", token="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", streaming=True)

# for x in ds:
#     print(x['name'])
#     break
