import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# -----------------------------
# Custom Dataset for Flowers
# -----------------------------
class FlowersDataset(Dataset):
    def __init__(self, images_dir, labels, transform=None):
        """
        Args:
            images_dir (str): Path to directory containing flower images.
            labels (np.ndarray): 1D array of labels (1-indexed).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))  # Ensure consistent order

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        # Convert 1-indexed label to 0-indexed
        label = int(self.labels[idx]) - 1
        return image, label

# -----------------------------
# Load Pretrained YOLOv5 and Prepare Backbone
# -----------------------------
# Download and load YOLOv5s (requires internet on first run)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Freeze all YOLOv5 parameters
for param in yolo_model.parameters():
    param.requires_grad = False

# YOLOv5 model (from torch.hub) has a .model attribute that is an nn.Sequential of layers.
# We remove the detection head (usually the last module) and use the remainder as a backbone.
# (This architecture may change over time so adjust slicing as needed.)
backbone = nn.Sequential(*list(yolo_model.model.children())[:-1])

# Pass a dummy image to determine the backbone’s output channel dimension.
dummy_input = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    features = backbone(dummy_input)
# features shape is (batch, channels, H, W)
C = features.shape[1]

# Define a classification head: global pooling + flatten + linear layer for 102 classes.
classifier_head = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(C, 102)  # Oxford Flowers 102 classes
)

# Combine backbone and classifier head into one model.
class YOLOv5Classifier(nn.Module):
    def __init__(self, backbone, classifier):
        super(YOLOv5Classifier, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

model = YOLOv5Classifier(backbone, classifier_head)

# -----------------------------
# Training Setup
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
# We train only the classifier head parameters.
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

# -----------------------------
# Data Preparation
# -----------------------------
# Define image transformations.
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),  # YOLOv5 default input size
    transforms.ToTensor(),
    # (Optional) Normalize using ImageNet stats if desired:
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load labels from imagelabels.mat.
# The .mat file usually contains a variable named 'labels'
mat = scipy.io.loadmat('imagelabels.mat')
labels = mat['labels'][0]  # Adjust key if necessary

# Create full dataset.
dataset = FlowersDataset('102flowers/jpg', labels, transform=data_transforms)

# Split indices for training and validation (80/20 split)
all_indices = np.arange(len(dataset))
train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# -----------------------------
# Training Loop
# -----------------------------
if __name__ == '__main__':
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (preds == targets).sum().item()
        val_accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Accuracy: {val_accuracy:.4f}")

    print("Training complete.")
