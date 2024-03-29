import os
import cv2
import json

class PBR_Material:
    def __init__(self, basecolor, height, normal, roughness, desc):
        self.basecolor = basecolor
        self.height = height
        self.normal = normal
        self.roughness = roughness
        self.desc = desc

        self.shape = basecolor.shape
        
    #Rotate the material by 90, 180, 270 degrees
    def rotate(self):
        rot90_base = cv2.rotate(self.basecolor, cv2.ROTATE_90_CLOCKWISE)
        rot90_height = cv2.rotate(self.height, cv2.ROTATE_90_CLOCKWISE)
        rot90_normal = cv2.rotate(self.normal, cv2.ROTATE_90_CLOCKWISE)
        rot90_roughness = cv2.rotate(self.roughness, cv2.ROTATE_90_CLOCKWISE)

        rot90 = PBR_Material(rot90_base, rot90_height, rot90_normal, rot90_roughness, desc)

        rot180_base = cv2.rotate(self.basecolor, cv2.ROTATE_180)
        rot180_height = cv2.rotate(self.height, cv2.ROTATE_180)
        rot180_normal = cv2.rotate(self.normal, cv2.ROTATE_180)
        rot180_roughness = cv2.rotate(self.roughness, cv2.ROTATE_180)

        rot180 = PBR_Material(rot180_base, rot180_height, rot180_normal, rot180_roughness, desc)

        rot270_base = cv2.rotate(self.basecolor, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rot270_height = cv2.rotate(self.height, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rot270_normal = cv2.rotate(self.normal, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rot270_roughness = cv2.rotate(self.roughness, cv2.ROTATE_90_COUNTERCLOCKWISE)

        rot270 = PBR_Material(rot270_base, rot270_height, rot270_normal, rot270_roughness, desc)

        return rot90, rot180, rot270
    
    #Flip the material horizontally
    def flip(self):
        flip_base = cv2.flip(self.basecolor, 1)
        flip_height = cv2.flip(self.height, 1)
        flip_normal = cv2.flip(self.normal, 1)
        flip_roughness = cv2.flip(self.roughness, 1)

        flip = PBR_Material(flip_base, flip_height, flip_normal, flip_roughness, desc)

        return flip
    
    #Resize the material to a new size
    def resize(self, size):
        resized_base = cv2.resize(self.basecolor, size)
        resized_height = cv2.resize(self.height, size)
        resized_normal = cv2.resize(self.normal, size)
        resized_roughness = cv2.resize(self.roughness, size)

        resized = PBR_Material(resized_base, resized_height, resized_normal, resized_roughness, desc)

        return resized
    
    #Only receives a 1024*1024 image and outputs 4 512*512 images
    def crop(self):
        if self.shape != (1024, 1024, 3):
            raise ValueError("Input image must be 1024x1024. Image Shape is ", self.shape)
     
        basecolor_topleft = self.basecolor[:512, :512]
        basecolor_topright = self.basecolor[:512, 512:]
        basecolor_bottomleft = self.basecolor[512:, :512]  
        basecolor_bottomright = self.basecolor[512:, 512:]

        height_topleft = self.height[:512, :512]
        height_topright = self.height[:512, 512:]
        height_bottomleft = self.height[512:, :512]
        height_bottomright = self.height[512:, 512:]

        normal_topleft = self.normal[:512, :512]
        normal_topright = self.normal[:512, 512:]
        normal_bottomleft = self.normal[512:, :512]
        normal_bottomright = self.normal[512:, 512:]

        roughness_topleft = self.roughness[:512, :512]
        roughness_topright = self.roughness[:512, 512:]
        roughness_bottomleft = self.roughness[512:, :512]
        roughness_bottomright = self.roughness[512:, 512:]

        topleft = PBR_Material(basecolor_topleft, height_topleft, normal_topleft, roughness_topleft, desc)
        topright = PBR_Material(basecolor_topright, height_topright, normal_topright, roughness_topright, desc)
        bottomleft = PBR_Material(basecolor_bottomleft, height_bottomleft, normal_bottomleft, roughness_bottomleft, desc)
        bottomright = PBR_Material(basecolor_bottomright, height_bottomright, normal_bottomright, roughness_bottomright, desc)
        
        return topleft, topright, bottomleft, bottomright
    
    def save(self, path, name):
        os.makedirs(os.path.join(path, name), exist_ok=True)
        
        cv2.imwrite(os.path.join(path, name, "basecolor.png"), self.basecolor)
        cv2.imwrite(os.path.join(path, name, "height.png"), self.height)
        cv2.imwrite(os.path.join(path, name, "normal.png"), self.normal)
        cv2.imwrite(os.path.join(path, name, "roughness.png"), self.roughness)

        with open(os.path.join(path, name, "desc.json"), "w") as file:
            file.write(self.desc)

def desc_to_json(idx):
    with open(os.path.join("data/train", idx, "desc.txt"), 'r') as file:
        data = file.readlines()
    
    tags = data[0].strip().split(": ")[1].split(", ")
    name = data[1].strip().split(": ")[1]

    desc = {
        "Name": name,
        "Tags": tags,
        "Index": int(idx)
    }

    json_string = json.dumps(desc, indent=4)

    return json_string

og_path = "data/train"
new_path = "data/transformed"

dataset_size = 5699
count = 0

for i in range(0, dataset_size+1):

    #Get original material from file
    basecolor = cv2.imread(os.path.join(og_path, str(i), "basecolor.png"), cv2.IMREAD_UNCHANGED)
    height = cv2.imread(os.path.join(og_path, str(i), "height.png"), cv2.IMREAD_UNCHANGED)
    normal = cv2.imread(os.path.join(og_path, str(i), "normal.png"), cv2.IMREAD_UNCHANGED)
    roughness = cv2.imread(os.path.join(og_path, str(i), "roughness.png"), cv2.IMREAD_UNCHANGED)
    desc = desc_to_json(str(i))

    #Create PBR_Material object
    material = PBR_Material(basecolor, height, normal, roughness, desc)

    #Rotate original mateiral by 90, 180, 270
    rot90, rot180, rot270 = material.rotate()

    #Crop each rotation into 4 512*512 images
    rot90_left, rot90_right, rot90_top, rot90_bottom = rot90.crop()
    rot180_left, rot180_right, rot180_top, rot180_bottom = rot180.crop()
    rot270_left, rot270_right, rot270_top, rot270_bottom = rot270.crop()

    #Flip original material
    material_flip = material.flip()

    #Crop each rotation into 2 512*512 images
    rot90_flip_left, rot90_flip_right, rot90_flip_top, rot90_flip_bottom = material_flip.crop()

    #Save original material resize
    material_resized = material.resize((512, 512))

    # Save all images
    material_resized.save(new_path, str(count))
    
    rot90_left.save(new_path, str(count+1))
    rot90_right.save(new_path, str(count+2))
    rot90_top.save(new_path, str(count+3))
    rot90_bottom.save(new_path, str(count+4))

    rot180_left.save(new_path, str(count+5))
    rot180_right.save(new_path, str(count+6))
    rot180_top.save(new_path, str(count+7))
    rot180_bottom.save(new_path, str(count+8))

    rot270_left.save(new_path, str(count+9))
    rot270_right.save(new_path, str(count+10))
    rot270_top.save(new_path, str(count+11))
    rot270_bottom.save(new_path, str(count+12))

    rot90_flip_left.save(new_path, str(count+13))
    rot90_flip_right.save(new_path, str(count+14))
    rot90_flip_top.save(new_path, str(count+15))
    rot90_flip_bottom.save(new_path, str(count+16))

    count += 17

    print(f"Processed {i}/{dataset_size}")

    









        
    


