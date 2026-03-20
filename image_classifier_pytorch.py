# ============================================================
#  Project 3 — AI Image Classifier (PyTorch Version)
#  AI/ML Internship — LMS Trainee Program
#  Submitted by: Uttam Kumar
#  Dataset     : CIFAR-10 | Framework: PyTorch
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42); np.random.seed(42)

CLASS_NAMES = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

print("="*55)
print("  CIFAR-10 CNN Classifier — Uttam Kumar (PyTorch)")
print("="*55)

# ── 1. Data ───────────────────────────────────────────────────
print("\n[1/6] Loading CIFAR-10 Dataset...")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

full_train = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
test_set   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

val_size   = int(0.2 * len(full_train))
train_size = len(full_train) - val_size
train_set, val_set = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False, num_workers=2)

print(f"  Train: {train_size} | Val: {val_size} | Test: {len(test_set)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

# ── 2. Model ──────────────────────────────────────────────────
print("\n[2/6] Building CNN Model...")

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a=nn.Conv2d(3,32,3,padding=1);  self.conv1b=nn.Conv2d(32,32,3,padding=1)
        self.bn1a=nn.BatchNorm2d(32);             self.bn1b=nn.BatchNorm2d(32)
        self.conv2a=nn.Conv2d(32,64,3,padding=1); self.conv2b=nn.Conv2d(64,64,3,padding=1)
        self.bn2a=nn.BatchNorm2d(64);             self.bn2b=nn.BatchNorm2d(64)
        self.conv3a=nn.Conv2d(64,128,3,padding=1);self.conv3b=nn.Conv2d(128,128,3,padding=1)
        self.bn3a=nn.BatchNorm2d(128);            self.bn3b=nn.BatchNorm2d(128)
        self.fc1=nn.Linear(128*4*4,256); self.bn_fc=nn.BatchNorm1d(256); self.fc2=nn.Linear(256,10)
        self.pool=nn.MaxPool2d(2,2); self.drop25=nn.Dropout(0.25); self.drop50=nn.Dropout(0.5)

    def forward(self,x):
        x=F.relu(self.bn1a(self.conv1a(x))); x=F.relu(self.bn1b(self.conv1b(x))); x=self.pool(x); x=self.drop25(x)
        x=F.relu(self.bn2a(self.conv2a(x))); x=F.relu(self.bn2b(self.conv2b(x))); x=self.pool(x); x=self.drop25(x)
        x=F.relu(self.bn3a(self.conv3a(x))); x=F.relu(self.bn3b(self.conv3b(x))); x=self.pool(x); x=self.drop25(x)
        x=x.view(x.size(0),-1); x=F.relu(self.bn_fc(self.fc1(x))); x=self.drop50(x)
        return self.fc2(x)

model = CIFAR10_CNN().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,}")

# ── 3. Training ───────────────────────────────────────────────
print("\n[3/6] Training Model (30 epochs)...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_val_acc = 0.0
patience = 7
wait = 0
history = {'train_acc':[], 'val_acc':[], 'train_loss':[], 'val_loss':[]}

for epoch in range(30):
    # Train
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(labels).sum().item()
        train_total += labels.size(0)

    # Validate
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    t_acc = train_correct / train_total
    v_acc = val_correct / val_total
    t_loss = train_loss / len(train_loader)
    v_loss = val_loss / len(val_loader)

    history['train_acc'].append(t_acc)
    history['val_acc'].append(v_acc)
    history['train_loss'].append(t_loss)
    history['val_loss'].append(v_loss)

    scheduler.step()
    print(f"  Epoch {epoch+1:02d}/30 | Train: {t_acc*100:.1f}% | Val: {v_acc*100:.1f}% | Loss: {v_loss:.4f}")

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), 'best_model.pth')
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

# ── 4. Plots ──────────────────────────────────────────────────
print("\n[4/6] Saving Training Plots...")
fig, axes = plt.subplots(1,2,figsize=(13,4))
fig.suptitle("Training History — Uttam Kumar | CIFAR-10 CNN", fontweight='bold')
axes[0].plot(history['train_acc'], label='Train', color='#1565C0', lw=2)
axes[0].plot(history['val_acc'],   label='Val',   color='#0288D1', lw=2, ls='--')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history['train_loss'], label='Train', color='#B71C1C', lw=2)
axes[1].plot(history['val_loss'],   label='Val',   color='#E53935', lw=2, ls='--')
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('training_history.png', dpi=150); plt.close()
print("  Saved: training_history.png")

# ── 5. Evaluate ───────────────────────────────────────────────
print("\n[5/6] Evaluating on Test Set...")
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

all_preds, all_labels = [], []
correct = total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = correct / total
print(f"\n  ✔ Test Accuracy: {test_acc*100:.2f}%")
print("\n  Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix — Uttam Kumar | CIFAR-10", fontweight='bold')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=150); plt.close()
print("  Saved: confusion_matrix.png")

# ── 6. Summary ────────────────────────────────────────────────
print("\n"+"="*55)
print("  PROJECT COMPLETE")
print("="*55)
print(f"  Student    : Uttam Kumar")
print(f"  Internship : LMS Trainee Program (via LinkedIn)")
print(f"  Dataset    : CIFAR-10 | Framework: PyTorch")
print(f"  Test Acc   : {test_acc*100:.2f}%")
print(f"  Model Saved: best_model.pth")
print("="*55)
print("\n  ✔ best_model.pth — upload this to GitHub!")
print("  ✔ training_history.png")
print("  ✔ confusion_matrix.png")
