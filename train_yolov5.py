import os
import shutil
import yaml
import torch
import scipy.io
from sklearn.model_selection import train_test_split
from yolov5 import train  # Ensure the YOLOv5 repository is in your PYTHONPATH


# -----------------------------
# Step 1: Generate YOLO-format Annotations
# -----------------------------
def generate_yolo_annotations(images_dir, annotations_dir, labels):
    """
    For each image, create a YOLO-format annotation file.
    Assumes each image contains one object (the flower) that fills the image.
    Annotation: <class> 0.5 0.5 1 1
    """
    os.makedirs(annotations_dir, exist_ok=True)
    image_files = sorted(os.listdir(images_dir))
    if len(image_files) != len(labels):
        raise ValueError("Number of images and labels do not match!")
    for idx, img_file in enumerate(image_files):
        label = int(labels[idx]) - 1  # convert to 0-indexed
        annotation_line = f"{label} 0.5 0.5 1 1\n"
        base_name = os.path.splitext(img_file)[0]
        out_path = os.path.join(annotations_dir, base_name + ".txt")
        with open(out_path, "w") as f:
            f.write(annotation_line)
    print(f"Annotations created in {annotations_dir}")


# -----------------------------
# Step 2: Organize Dataset into YOLOv5 Structure
# -----------------------------
def organize_dataset(source_images_dir, source_labels_dir, base_dir, test_size=0.25, random_state=42):
    """
    Split images into train/val/test and copy images and corresponding annotation files
    into the YOLOv5 expected folder structure under base_dir.
    """
    image_files = sorted(os.listdir(source_images_dir))

    # Split file names
    train_val, test = train_test_split(image_files, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train_val, test_size=1 / 3, random_state=random_state)

    # Define destination directories
    images_train_dir = os.path.join(base_dir, "images", "train")
    images_val_dir = os.path.join(base_dir, "images", "val")
    images_test_dir = os.path.join(base_dir, "images", "test")
    labels_train_dir = os.path.join(base_dir, "labels", "train")
    labels_val_dir = os.path.join(base_dir, "labels", "val")
    labels_test_dir = os.path.join(base_dir, "labels", "test")

    for d in [images_train_dir, images_val_dir, images_test_dir,
              labels_train_dir, labels_val_dir, labels_test_dir]:
        os.makedirs(d, exist_ok=True)

    def copy_files(file_list, dest_img_dir, dest_lbl_dir):
        for file in file_list:
            # Copy image file
            shutil.copy2(os.path.join(source_images_dir, file), dest_img_dir)
            # Copy corresponding annotation file (assumes same base name with .txt)
            base = os.path.splitext(file)[0]
            label_file = base + ".txt"
            src_label = os.path.join(source_labels_dir, label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dest_lbl_dir)
            else:
                print(f"Warning: Label file {src_label} not found.")

    copy_files(train, images_train_dir, labels_train_dir)
    copy_files(val, images_val_dir, labels_val_dir)
    copy_files(test, images_test_dir, labels_test_dir)

    print(f"Dataset organized: train({len(train)}), val({len(val)}), test({len(test)})")
    return images_train_dir, images_val_dir  # YOLOv5 uses train and val for training


# -----------------------------
# Step 3: Create data.yaml for YOLOv5
# -----------------------------
def create_data_yaml(train_dir, val_dir, nc, names, yaml_path):
    data = {
        "train": os.path.abspath(train_dir),
        "val": os.path.abspath(val_dir),
        "nc": nc,
        "names": names
    }
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"data.yaml created at {os.path.abspath(yaml_path)}")
    return yaml_path


# -----------------------------
# Main: Prepare Dataset and Run Training
# -----------------------------
if __name__ == "__main__":
    # Define source directories (adjust if needed)
    source_images_dir = "102flowers/jpg"  # your original images folder
    source_labels_dir = "102flowers/labels"  # annotations will be generated here
    base_dir = "102flowers"  # base folder for new structure

    # Load labels from imagelabels.mat
    mat = scipy.io.loadmat("imagelabels.mat")
    labels = mat["labels"][0]

    # Generate annotations in source_labels_dir (if not already generated)
    generate_yolo_annotations(source_images_dir, source_labels_dir, labels)

    # Organize dataset into YOLOv5 structure: images/train, images/val, images/test and corresponding labels
    images_train_dir, images_val_dir = organize_dataset(source_images_dir, source_labels_dir, base_dir)

    # Create data.yaml file for YOLOv5 training
    nc = 102
    names = [f"flower_{i}" for i in range(nc)]
    yaml_path = os.path.join(base_dir, "data.yaml")
    create_data_yaml(images_train_dir, images_val_dir, nc, names, yaml_path)

    # -----------------------------
    # Step 4: Define Training Parameters and Run YOLOv5 Training
    # -----------------------------
    command_dict = {
        "data": yaml_path,
        "weights": "yolov5/yolov5s.pt",  # pretrained weights from YOLOv5
        "epochs": 30,
        "batch": 16,
        "imgsz": 640,
        "freeze": [20],  # freeze first 10 layers (as a list)
        "optimizer": "AdamW",  # use AdamW optimizer
        "cos_lr": True,  # cosine learning rate scheduling
        "label_smoothing": 0.1,  # label smoothing
        "multi_scale": True,  # enable multi-scale training
        "patience": 5,
        "project": "training_batches",
        "name": "train",
        "device": "",  # auto-select GPU/CPU
        "workers": 6,
        "save_period": 0,
        "cache": True
    }

    # Run YOLOv5 training
    train.run(**command_dict)
