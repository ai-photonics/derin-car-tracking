import glob
import random
import os
from pathlib import Path
import shutil
from PIL import Image


class Meta:

    # definition of image metadata for annotated images
    def __init__(self, id, img_path, label_path, size, n_objects, tags):
        self.id = id
        self.img_path = img_path
        self.label_path = label_path
        self.size = size
        self.n_objects = n_objects
        self.tags = tags

    def __str__(self):
        return f"Meta({self.id},{self.tags})"


def convert2yolo_file(label_file, target_label_file, img_file):
        """This function is adapted from the YOLO official conversion function"""
        img_size = Image.open(img_file).size
        dw, dh = 1.0 / img_size[0], 1.0 / img_size[1]
        lines = []

        with open(label_file, encoding="utf-8") as file:
            for row in [x.split(",") for x in file.read().strip().splitlines()]:
                if row[4] != "0":  # Skip ignored regions
                    x, y, w, h = map(int, row[:4])
                    cls = int(row[5]) - 1
                    # Convert to YOLO format
                    x_center, y_center = (x + w / 2) * dw, (y + h / 2) * dh
                    w_norm, h_norm = w * dw, h * dh
                    lines.append(
                        f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                    )
        # Write the label in YOLO format to the target file
        target_label_file.write_text("".join(lines), encoding="utf-8")


def copy_temp_data(base_path, target_subfolder, split_name, ids):
    image_path = base_path / target_subfolder / split_name / "images" 
    label_path = base_path / target_subfolder / split_name / "labels" 
    image_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)
    for id in ids:
        print(f"Copying {id}.jpg to {image_path}")
        shutil.copy((source_image_path / f"{id}.jpg"), (image_path / f"{id}.jpg"))
        print(f"Copying {id}.txt to {label_path}")
        convert2yolo_file(
            label_file=(source_label_path / f"{id}.txt"), 
            target_label_file=(label_path / f"{id}.txt"), 
            img_file=(source_image_path / f"{id}.jpg")
        )
        #shutil.copy(, (label_path / f"{id}.txt"))


def get_ori_data(image_paths_ori, tags, label_path_ori):
    img_metas = []
    for i, image_path_ori in enumerate(image_paths_ori):
        img_files = glob.glob('*.jpg', root_dir=image_path_ori)
        img_ids = [img_file.replace('.jpg', '') for img_file in img_files]
        img_metas_tmp = [Meta(id=img_id, img_path="", label_path="", size=0, n_objects=0, tags=tags[i]) for img_id in img_ids]
    img_metas.append(img_metas_tmp)

    return img_metas


def split_data(img_ids, split_ratios):
    n = len(img_ids)
    split_n = [int(n * split_ratio) for split_ratio in split_ratios]
    split_n[2] = n - split_n[0] - split_n[1]
    rand_ids = random.sample(img_ids, k=n)
    train_ids = rand_ids[0:split_n[0]]
    val_ids = rand_ids[split_n[0]:(split_n[0]+split_n[1])]
    test_ids = rand_ids[(split_n[0]+split_n[1]):n]
    
    return train_ids, val_ids, test_ids


def select_data(img_meta_list, ns):
    ids = list(range(len(img_meta_list)))
    rand_ids = random.sample(ids, k=ns)
    img_meta_sub = img_meta_list[rand_ids]
    path_sub = [meta.img_path for meta in img_meta_sub]

    return path_sub, img_meta_sub


def list_image_ids(input_folder):
    img_files = glob.glob('*.jpg', root_dir=input_folder)
    img_ids = [img_file.replace('.jpg', '') for img_file in img_files]
    return img_ids, img_files


def select_files(input_folder, output_paths, split_ns, do_append=False):
    img_ids, img_files = list_image_ids(input_folder / "images")
    # Shuffle all samples first to prevent bias in train/valid/test selection
    random.shuffle(img_files)
    start_n = 0
    stop_n = 0
    if do_append: 
        write_mode = 'a+' 
    else: 
        write_mode = 'w+'
    for j, n in enumerate(split_ns):
        stop_n = stop_n + n
        img_paths = [str(input_folder / "images") + "/" + img_file + "\n" for img_file in sorted(img_files[start_n:stop_n])]
        with open(output_paths[j], mode=write_mode) as f:
            f.writelines(img_paths)
        start_n = start_n + n


def convert_labels(input_folder, source_label_path):
    img_ids, img_files = list_image_ids(input_folder / "images")
    label_path = input_folder / "labels"
    for id in sorted(img_ids):
        convert2yolo_file(
            label_file=(source_label_path / f"{id}.txt"), 
            target_label_file=(label_path / f"{id}.txt"), 
            img_file=(input_folder / "images" / f"{id}.jpg")
        )


#base_path = Path("./data/visdrone/det")
#source_image_path = base_path / "dark"
#source_label_path = base_path / "annotations"
#
#image_paths_ori = [(base_path / "dark"), (base_path / "light")]
#print(image_paths_ori)
#label_path_ori = base_path / "annotations"
#
#
#img_metas_list = get_ori_data(image_paths_ori=image_paths_ori, tags=["dark", "light"], label_path_ori=source_label_path)
#[print(img_meta) for img_meta in img_metas_list[1]]
# Get all filenames

# Split data
#split_ratios = [0.7, 0.2, 0.1]

image_type = "light-50_dark-50"
base_path = Path("./data/visdrone/det")
input_folders = [(base_path / "light"), (base_path / "dark")]
output_paths = [
    (base_path / f"{image_type}set_train.txt"), 
    (base_path / f"{image_type}set_valid.txt"), 
    (base_path / f"{image_type}set_test.txt")
]
source_label_path = base_path / "annotations"
for input_folder in input_folders:
    select_files(input_folder, output_paths, split_ns=[50, 32, 30], do_append=True)
    convert_labels(input_folder, source_label_path)

## Temporarily copy data to the images/annotations folder
#copy_temp_data(base_path, split_name="train", ids=train_ids)
#copy_temp_data(base_path, split_name="valid", ids=val_ids)
#copy_temp_data(base_path, split_name="test", ids=test_ids)    

#results = model.train(data="data_det_light.yaml", epochs=3, imgsz=640)
