import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# ===============================
# Config
# ===============================
path = 'ai-vs-human-generated-dataset'
train_csv = 'detect-ai-vs-human-generated-images/train.csv'
test_csv = 'detect-ai-vs-human-generated-images/test.csv'
batch_size = 32
epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Dataset Setup
# ===============================
train_df = pd.read_csv(train_csv)[['file_name', 'label']]
train_df.columns = ['id', 'label']
test_df = pd.read_csv(test_csv)

train_split, val_split = train_test_split(train_df, test_size=0.05, stratify=train_df['label'], random_state=42)

# ===============================
# Transforms
# ===============================
train_tfms = transforms.Compose([
    transforms.Resize(232),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===============================
# Dataset Class
# ===============================
class AIImageDataset(Dataset):
    def __init__(self, df, root, transform, has_label=True):
        self.df = df
        self.root = root
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.df.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.has_label:
            label = int(self.df.iloc[idx, 1])
            return image, label
        return image, self.df.iloc[idx, 0]  # return id

# ===============================
# Dataloaders
# ===============================
train_loader = DataLoader(AIImageDataset(train_split, path, train_tfms), batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(AIImageDataset(val_split, path, val_tfms), batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(AIImageDataset(test_df, path, val_tfms, has_label=False), batch_size=batch_size, shuffle=False)

# ===============================
# Model
# ===============================
model = models.convnext_base(weights="DEFAULT")

for param in model.features.parameters():
    param.requires_grad = False
for param in model.features[-2:].parameters():
    param.requires_grad = True

model.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)
)
model.to(device)

# ===============================
# Training Setup
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([
    {'params': model.features[-2:].parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
])
scheduler = StepLR(optimizer, step_size=5, gamma=0.7)

# ===============================
# Training Loop
# ===============================
def evaluate(loader):
    model.eval()
    val_loss, preds, targets = 0, [], []
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            val_loss += loss.item()
            preds.extend(output.argmax(1).cpu().numpy())
            targets.extend(label.cpu().numpy())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    return val_loss / len(loader), acc, f1

for epoch in range(epochs):
    model.train()
    total_loss, total_acc = 0, 0
    for data, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += (output.argmax(1) == label).float().mean().item()

    val_loss, val_acc, val_f1 = evaluate(val_loader)
    scheduler.step()

    print(f"Epoch {epoch+1}: Train Loss {total_loss/len(train_loader):.4f}, "
          f"Train Acc {total_acc/len(train_loader):.4f}, "
          f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Val F1 {val_f1:.4f}")

# ===============================
# Save the Trained Model
# ===============================
save_path = "working/convnext_ai_vs_human.pth"
torch.save(model.state_dict(), save_path)