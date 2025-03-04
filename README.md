# Flower Classification and Detection Project

This repository contains code for a multi-model flower classification/detection pipeline using two different approaches:

1. **VGG19 Model (TensorFlow/Keras):**  
   A VGG19-based classifier trained to classify 102 types of flowers. The pipeline includes data preparation, training, evaluation, and plotting of accuracy and loss curves.

2. **YOLOv5 Model (PyTorch):**  
   A YOLOv5-based model (adapted for flower classification/detection) that converts raw flower images into YOLO-format annotations, organizes the data into the required folder structure, and trains using YOLOv5's full pipeline. Training metrics (including mAP@0.5 and classification loss) are plotted and saved automatically.

## Repository Structure

- **main.py**  
  The main entry point that sequentially calls the data preparation, VGG19 training, and YOLOv5 training scripts.

- **vgg_data_prep.py**  
  A module for splitting and organizing the data for the VGG19 model. It creates CSV files for train, validation, and test splits.

- **train_vgg19.py**  
  Contains the code for training a VGG19 classifier using TensorFlow/Keras. It:
  - Loads data splits from CSV files.
  - Uses data augmentation via `ImageDataGenerator`.
  - Constructs a VGG19-based model with additional dense layers and dropout.
  - Trains the model and evaluates it on a test set.
  - Plots and saves the training and validation accuracy and loss curves.

- **train_yolov5.py**  
  Contains the code for training a YOLOv5 model. It:
  - Generates YOLO-format annotations from raw labels.
  - Organizes images and annotations into the folder structure expected by YOLOv5.
  - Creates a `data.yaml` file that defines the dataset (with absolute paths).
  - Calls YOLOv5’s built-in training pipeline with specified parameters.
  - After training, it reads the results CSV file, plots training metrics (e.g., validation mAP@0.5 and cross-entropy losses), and saves the plots in a folder (`yolo_plots`).

## Requirements

- **Python 3.8+**
- **TensorFlow (for VGG19 training)**
- **PyTorch (for YOLOv5 training)**
- **OpenCV (cv2)**
- **Pandas, NumPy, scikit-learn**
- **Matplotlib, PyYAML**

Install dependencies via pip:

```bash
pip install tensorflow torch opencv-python pandas numpy scikit-learn matplotlib pyyaml
