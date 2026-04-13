"""
Dog Breed Prediction — Evaluation Script
Generates classification report + confusion matrix (confusion_matrix.png)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_DIR   = "dataset_split"
MODEL_PATH = "model.pth"
BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Load val data
# ─────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "val"),
    transform=val_transform
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]

model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, num_classes),
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────
print("\n📊 Classification Report")
print("─" * 60)
print(classification_report(all_labels, all_preds, target_names=class_names))

acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"✅ Overall Accuracy: {acc*100:.2f}%")

# ─────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="YlOrBr",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor="white",
)
plt.title("Dog Breed Prediction — Confusion Matrix", fontsize=16, pad=15)
plt.ylabel("Actual Breed", fontsize=12)
plt.xlabel("Predicted Breed", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("\n🖼  Confusion matrix saved → confusion_matrix.png")
