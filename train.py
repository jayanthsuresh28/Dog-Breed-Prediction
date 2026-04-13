"""
Dog Breed Prediction — Training Script
Model  : EfficientNet-B0 (Transfer Learning via PyTorch)
Dataset: 10 breeds × 100 images each (custom, placed in dataset_split/)
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_DIR   = "dataset_split"   # train/ and val/ subdirs
MODEL_PATH = "model.pth"
NUM_CLASSES = 10
BATCH_SIZE  = 16
NUM_EPOCHS  = 20
LR          = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BREEDS = [
    "Beagle", "Boxer", "Bulldog", "Dachshund", "German_Shepherd",
    "Golden_Retriever", "Labrador_Retriever", "Poodle", "Rottweiler", "Yorkshire_Terrier"
]

# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}

# ─────────────────────────────────────────────
# Dataset & DataLoaders
# ─────────────────────────────────────────────
image_datasets = {
    phase: datasets.ImageFolder(
        root=os.path.join(DATA_DIR, phase),
        transform=data_transforms[phase]
    )
    for phase in ["train", "val"]
}

dataloaders = {
    phase: DataLoader(
        image_datasets[phase],
        batch_size=BATCH_SIZE,
        shuffle=(phase == "train"),
        num_workers=2,
    )
    for phase in ["train", "val"]
}

dataset_sizes = {phase: len(image_datasets[phase]) for phase in ["train", "val"]}
class_names   = image_datasets["train"].classes

print(f"Classes  : {class_names}")
print(f"Train sz : {dataset_sizes['train']}  |  Val sz: {dataset_sizes['val']}")
print(f"Device   : {DEVICE}")

# ─────────────────────────────────────────────
# Model — EfficientNet-B0
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model

model = build_model(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_weights = copy.deepcopy(model.state_dict())
    best_acc     = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}  {'─'*40}")
        t0 = time.time()

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            print(f"  {phase.upper():5s}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc     = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())

        print(f"  Time: {time.time()-t0:.1f}s")

    print(f"\n✅ Best Val Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_weights)
    return model


if __name__ == "__main__":
    model = train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names":      class_names,
        "num_classes":      NUM_CLASSES,
    }, MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")
