import glob
import random
import os
from pathlib import Path
import shutil


def copy_temp_data(base_path, split_name, ids):
    image_path = base_path / split_name / "images" 
    label_path = base_path / split_name / "labels" 
    image_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)
    for id in ids:
        print(f"Copying {id}.jpg to {image_path}")
        shutil.copy((source_image_path / f"{id}.jpg"), (image_path / f"{id}.jpg"))
        print(f"Copying {id}.txt to {label_path}")
        shutil.copy((source_label_path / f"{id}.txt"), (label_path / f"{id}.txt"))


# Get all filenames
base_path = Path("./data/visdrone/det")
source_image_path = base_path / "light"
source_label_path = base_path / "annotations"
img_files = glob.glob('*.jpg', root_dir='./data/visdrone/det/light')
img_ids = [img_file.replace('.jpg', '') for img_file in img_files]
print(img_files)
print(img_ids)

# Split data
n = len(img_ids)
split_ratios = [0.7, 0.2, 0.1]
split_n = [int(n * split_ratio) for split_ratio in split_ratios]
split_n[2] = n - split_n[0] - split_n[1]
rand_ids = random.sample(img_ids, k=n)
train_ids = rand_ids[0:split_n[0]]
val_ids = rand_ids[split_n[0]:(split_n[0]+split_n[1])]
test_ids = rand_ids[(split_n[0]+split_n[1]):n]

# Temporarily copy data to the images/annotations folder
copy_temp_data(base_path, split_name="train", ids=train_ids)
copy_temp_data(base_path, split_name="valid", ids=val_ids)
copy_temp_data(base_path, split_name="test", ids=test_ids)    

#results = model.train(data="data_det_light.yaml", epochs=3, imgsz=640)
