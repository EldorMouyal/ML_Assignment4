# vgg_data_prep.py

import os
import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split

def load_data(image_dir, labels_path):
    # Load labels from .mat file
    labels_data = scipy.io.loadmat(labels_path)
    labels = labels_data['labels'][0]  # Extract labels array

    # Get sorted list of image filenames
    image_files = sorted(os.listdir(image_dir))

    # Create DataFrame pairing each image with its label
    data = pd.DataFrame({
        'filename': image_files,
        'label': labels
    })

    return data

def split_data(data, test_size=0.25, val_size=0.333, random_state=42):
    # Initial split into train_val and test sets
    train_val_data, test_data = train_test_split(
        data,
        test_size=test_size,
        stratify=data['label'],
        random_state=random_state
    )

    # Further split train_val into training and validation sets
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size,
        stratify=train_val_data['label'],
        random_state=random_state
    )

    return train_data.reset_index(drop=True), val_data.reset_index(drop=True), test_data.reset_index(drop=True)

def adjust_labels(data):
    # Adjust labels to start from 0 and convert to strings
    data = data.copy()
    data['label'] = data['label'] - 1
    data['label'] = data['label'].astype(str)
    return data

def save_data_splits(train_data, val_data, test_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    print("Data splits saved to:", output_dir)

def main():
    # Paths to data
    image_dir = '102flowers/jpg'  # Adjust if necessary
    labels_path = 'imagelabels.mat'
    output_dir = 'data_splits'

    # Load data
    data = load_data(image_dir, labels_path)
    print(f"Total images: {len(data)}")

    # Split data
    train_data, val_data, test_data = split_data(data)
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

    # Adjust labels
    train_data = adjust_labels(train_data)
    val_data = adjust_labels(val_data)
    test_data = adjust_labels(test_data)  # Labels adjusted for consistency, though not used in testing

    # Save splits
    save_data_splits(train_data, val_data, test_data, output_dir)

if __name__ == '__main__':
    main()
