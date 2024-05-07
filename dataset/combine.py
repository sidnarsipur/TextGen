# Combine the dataset of the texture images


if __name__ == "__main__":
    import os
    import shutil
    os.makedirs("data/combined", exist_ok=True)
    
    for i in range(89):
        os.makedirs(f"data/combined/{i}", exist_ok=True)
        #ground-truths
        shutil.copy(f"data/test/{i}/basecolor.png", f"data/combined/{i}/gt-basecolor.png")
        shutil.copy(f"data/test/{i}/normal.png", f"data/combined/{i}/gt-normal.png")
        shutil.copy(f"data/test/{i}/roughness.png", f"data/combined/{i}/gt-roughness.png")
        
        # matforge-inference
        shutil.copy(f"data/matforge-inference/{i}/normal.png", f"data/combined/{i}/matforge-normal.png")
        shutil.copy(f"data/matforge-inference/{i}/roughness.png", f"data/combined/{i}/matforge-roughness.png")
        
        # controlnet-inference
        shutil.copy(f"data/controlnet-inference/{i}/normal.png", f"data/combined/{i}/controlnet-normal.png")
        shutil.copy(f"data/controlnet-inference/{i}/roughness.png", f"data/combined/{i}/controlnet-roughness.png")
        
        print(f"Combined {i}")
    