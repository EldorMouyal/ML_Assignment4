import os
import shutil
import yaml
import torch
import scipy.io
from sklearn.model_selection import train_test_split
from yolov5 import train  # Ensure the YOLOv5 repository is in your PYTHONPATH

import matplotlib.pyplot as plt
import pandas as pd


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
            shutil.copy2(os.path.join(source_images_dir, file), dest_img_dir)
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
# Step 4: Plotting Function for Metrics
# -----------------------------
def plot_training_results(csv_path, save_path="yolo_plots/training_curves.png"):
    """
    Plots side-by-side subplots:
      - Left: "accuracy" (mAP@0.5) vs. epoch
      - Right: classification loss (train vs. validation) vs. epoch

    Args:
        csv_path (str): Path to YOLOv5 results.csv file
        save_path (str): Where to save the output figure
    """

    # 1. Load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    # 2. Strip any leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()
    print("Columns found:", df.columns.tolist())

    # 3. Check for an 'epoch' column
    if "epoch" not in df.columns:
        print("No 'epoch' column found in the CSV.")
        return

    epochs = df["epoch"]

    # 4. Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---------------------------
    # Left Subplot: Accuracy (mAP@0.5) vs. Epoch
    # ---------------------------
    if "metrics/mAP_0.5" in df.columns:
        axes[0].plot(epochs, df["metrics/mAP_0.5"], marker='o', label="Validation Accuracy (mAP@0.5)")
    else:
        print("Column 'metrics/mAP_0.5' not found. Skipping accuracy plot.")

    axes[0].set_title("Accuracy vs. Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("mAP@0.5")
    axes[0].legend()
    axes[0].grid(True)

    # ---------------------------
    # Right Subplot: Loss (Train vs. Validation) vs. Epoch
    # ---------------------------
    # YOLOv5 typically logs train/cls_loss and val/cls_loss
    train_loss_col = "train/cls_loss"
    val_loss_col = "val/cls_loss"

    if train_loss_col in df.columns:
        axes[1].plot(epochs, df[train_loss_col], marker='o', label="Train Loss")
    else:
        print(f"Column '{train_loss_col}' not found. Skipping train loss plot.")

    if val_loss_col in df.columns:
        axes[1].plot(epochs, df[val_loss_col], marker='o', label="Validation Loss")
    else:
        print(f"Column '{val_loss_col}' not found. Skipping validation loss plot.")

    axes[1].set_title("Loss vs. Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    # 5. Finalize and save
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")


# -----------------------------
# Main: Prepare Dataset, Run Training, and Plot Metrics
# -----------------------------
if __name__ == "__main__":
    # Define source directories
    source_images_dir = "102flowers/jpg"  # your original images folder
    source_labels_dir = "102flowers/labels"  # annotations will be generated here
    base_dir = "102flowers"  # base folder for new structure

    # Load labels from imagelabels.mat
    mat = scipy.io.loadmat("imagelabels.mat")
    labels = mat["labels"][0]

    # Generate YOLO-format annotations
    generate_yolo_annotations(source_images_dir, source_labels_dir, labels)

    # Organize dataset into YOLOv5 structure
    images_train_dir, images_val_dir = organize_dataset(source_images_dir, source_labels_dir, base_dir)

    # Create data.yaml file for YOLOv5 training
    nc = 102
    names = [f"flower_{i}" for i in range(nc)]
    yaml_path = os.path.join(base_dir, "data.yaml")
    create_data_yaml(images_train_dir, images_val_dir, nc, names, yaml_path)

    # -----------------------------
    # Define Training Parameters and Run YOLOv5 Training
    # -----------------------------
    command_dict = {
        "data": yaml_path,
        "weights": "yolov5/yolov5s.pt",  # pretrained weights from YOLOv5
        "epochs": 30,
        "batch": 16,
        "imgsz": 640,
        "freeze": [15],  # freeze first 15 layers (as a list)
        "optimizer": "AdamW",  # use AdamW optimizer
        "cos_lr": True,  # cosine learning rate scheduling
        "label_smoothing": 0.1,  # label smoothing
        "multi_scale": True,  # enable multi-scale training
        "patience": 5,
        "project": "training_batches",
        "name": "train",
        "device": "",  # auto-select GPU/CPU
        "workers": 4,
        "save_period": -1,
        "cache": False
    }

    # Run YOLOv5 training
    train.run(**command_dict)

    # Define path to training_batches directory
    training_batches_dir = "training_batches"

    # Get the list of all train folders
    train_folders = [f for f in os.listdir(training_batches_dir) if f.startswith("train")]
    # Sort the train folders by their creation time to get the most recent one
    train_folders.sort(key=lambda x: os.path.getmtime(os.path.join(training_batches_dir, x)), reverse=True)
    # Get the most recent train folder
    latest_train_folder = train_folders[0]
    # Define the path to the results.csv file in the latest train folder
    results_file = os.path.join(training_batches_dir, latest_train_folder, "results.csv")
    plot_training_results(results_file)

