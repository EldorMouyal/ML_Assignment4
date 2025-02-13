# train_yolov5.py

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Add the YOLOv5 directory to sys.path
YOLOv5_PATH = os.path.join(os.path.dirname(__file__), 'yolov5')
if YOLOv5_PATH not in sys.path:
    sys.path.insert(0, YOLOv5_PATH)

from models.yolo import DetectionModel
from utils.torch_utils import select_device

# Custom classification model
class ClassificationModel(DetectionModel):
    def __init__(self, cfg, ch=3, nc=1000):
        super().__init__(cfg, ch=ch, nc=nc)

        # Remove the Detect() layer
        self.model = self.model[:-1]

        # Add classification head
        last_layer = self.model[-1]
        if hasattr(last_layer, 'cv2'):
            last_layer_out_channels = last_layer.cv2.out_channels
        elif hasattr(last_layer, 'cv1'):
            last_layer_out_channels = last_layer.cv1.out_channels
        elif hasattr(last_layer, 'conv'):
            last_layer_out_channels = last_layer.conv.out_channels
        else:
            # Use a dummy input to determine the output channels
            dummy_input = torch.zeros(1, ch, 224, 224)
            with torch.no_grad():
                x = dummy_input
                for m in self.model:
                    x = m(x)
                last_layer_out_channels = x.shape[1]

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_layer_out_channels, nc)
        )

    def forward(self, x):
        y = []  # outputs

        for m in self.model:
            if m.f != -1:  # if not first layer, get input from previous layers
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)

        x = self.fc(x)
        return x

# Dataset class
class FlowerDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.labels = self.data['label'].astype(int).values  # Labels already adjusted

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx]['filename'])
        image = Image.open(img_name).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        label = self.labels[idx]
        return image, label

# Function to create the classification model
def create_yolov5_classification_model(num_classes):
    # Paths to YOLOv5 configuration and weights
    cfg = os.path.join(YOLOv5_PATH, 'models', 'yolov5s.yaml')  # Use the detection model configuration
    weights = os.path.join(YOLOv5_PATH, 'yolov5s.pt')

    # Initialize the custom classification model
    model = ClassificationModel(cfg, ch=3, nc=num_classes)

    # Load pre-trained weights
    ckpt = torch.load(weights, map_location='cpu')
    state_dict = ckpt['model'].float().state_dict()
    # Exclude the Detect layer's weights
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.24')}
    model.load_state_dict(filtered_state_dict, strict=False)

    return model

# Training and evaluation function
def run_yolov5(train_data, val_data, test_data, image_dir, output_dir, num_classes=102, batch_size=32, epochs=10):
    # Image size
    img_size = 224

    # Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Datasets
    train_dataset = FlowerDataset(train_data, image_dir, transforms=data_transforms['train'])
    val_dataset = FlowerDataset(val_data, image_dir, transforms=data_transforms['val'])
    test_dataset = FlowerDataset(test_data, image_dir, transforms=data_transforms['val'])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Device
    device = select_device('')  # Use GPU if available

    # Model
    model = create_yolov5_classification_model(num_classes)
    model = model.to(device)

    # Freeze the pre-trained layers (optional)
    for param in model.model.parameters():
        param.requires_grad = False
    # Ensure classification head is trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # Forward pass

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total * 100
        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%")

        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_yolov5_model.pth'))

    print("Training completed. Best Validation Accuracy: {:.2f}%".format(best_val_acc))

    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_yolov5_model.pth')))

    # Testing
    model.eval()
    probabilities = []
    predictions = []
    filenames = test_data['filename'].tolist()

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            probs = nn.functional.softmax(outputs, dim=1)
            probabilities.append(probs.cpu().numpy()[0])
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.cpu().numpy()[0] + 1)  # Adjust back to original labels

    test_probabilities = pd.DataFrame({
        'filename': filenames,
        'probabilities': probabilities,
        'predicted_class': predictions
    })

    # Save predictions
    test_probabilities.to_csv(os.path.join(output_dir, 'yolov5_test_predictions.csv'), index=False)
    print("YOLOv5 training and prediction completed. Outputs saved to:", output_dir)

def main():
    # Paths
    split_dir = 'data_splits'
    image_dir = '102flowers/jpg'  # Adjust if necessary
    output_dir = 'yolov5_output'

    os.makedirs(output_dir, exist_ok=True)

    # Load data splits
    train_data = pd.read_csv(os.path.join(split_dir, 'train_data.csv'))
    val_data = pd.read_csv(os.path.join(split_dir, 'val_data.csv'))
    test_data = pd.read_csv(os.path.join(split_dir, 'test_data.csv'))

    # Ensure labels are integers
    for df in [train_data, val_data, test_data]:
        df['label'] = df['label'].astype(int) - 1  # Adjust labels to start from 0

    # Number of classes
    num_classes = len(train_data['label'].unique())

    # Run YOLOv5 classification
    run_yolov5(train_data, val_data, test_data, image_dir, output_dir, num_classes=num_classes, epochs=10)

if __name__ == '__main__':
    main()
