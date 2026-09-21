"""
Train / fine-tune ResNet-50 on the Face Mask Dataset from Kaggle.

Dataset layout expected (created automatically by download_dataset.py):
    dataset/
        with_mask/      ← images of people wearing a mask
        without_mask/   ← images of people NOT wearing a mask

Usage:
    python train.py
"""

import os, random, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = "dataset"
OUTPUT_PATH = "models/resnet_mask.pth"
EPOCHS      = 10
BATCH_SIZE  = 32
LR          = 1e-4
VAL_SPLIT   = 0.2
SEED        = 42
IMG_SIZE    = 224

torch.manual_seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ── Transforms ───────────────────────────────────────────────────────────────
# 流水线 Compose：把多步串成一条链，读图时自动按顺序执行
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),

    # 把像素除以 255，从 [0, 255] 归一化到 [0, 1]  tensor = img.float() / 255.0
    transforms.ToTensor(),

    # 保持这组官方数 对齐预训练模型的输入约定 
    # 像素值先归一化到 $[0, 1]$，再使用 ImageNet 数据集的均值与标准差进行标准化
    # ImageNet 这 120 万张图统计出来的均值和方差做标准化的。卷积核已经习惯这种分布
    # 均值：R 0.485，G 0.456，B 0.406  标准差：大约 0.22~0.23
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Dataset ──────────────────────────────────────────────────────────────────
def get_loaders():
    full = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    n_val = int(len(full) * VAL_SPLIT)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val])

    # Apply val transform to validation split
    val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[INFO] Classes : {full.classes}")
    print(f"[INFO] Train   : {n_train} images  |  Val: {n_val} images")
    return train_loader, val_loader, full.classes


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze all layers except layer4 and fc
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    if not os.path.isdir(DATA_DIR):
        print(f"[ERROR] Dataset folder '{DATA_DIR}' not found.")
        print("        Run  python download_dataset.py  first.")
        return

    train_loader, val_loader, classes = get_loaders()
    model     = build_model(len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct      += (out.argmax(1) == labels).sum().item()
            total        += imgs.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
              f"Loss: {train_loss:.4f}  "
              f"Train Acc: {train_acc:.2%}  "
              f"Val Acc: {val_acc:.2%}  "
              f"({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"  ✓ Saved best model  (val acc {val_acc:.2%})")

    print(f"\n[DONE] Best validation accuracy: {best_val_acc:.2%}")
    print(f"       Weights saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    train()
