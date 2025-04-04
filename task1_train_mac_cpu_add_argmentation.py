import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === 1. Dataset 類別 ===
class CTClassificationDataset(Dataset):
    def __init__(self, csv_file, image_root, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder = f"image{int(row['filename'][:4])}"   # e.g. image123
        img_path = self.image_root / folder / row["filename"]
        if not img_path.exists():
            print(f"❌ 找不到圖片：{img_path}")
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

# === 2. 訓練流程（CPU優化 + 驗證曲線 + EarlyStopping + 資料增強） ===
def train_task1_classification():
    csv_path = "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task1_classification_labels.csv"
    image_root = "/Users/chia-huitsao/Downloads/PHE-SICH-CT-IDS/SubdatasetC_PNG/set"

    batch_size = 4
    num_epochs = 20
    patience = 5  # Early stopping
    device = "cpu"

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)

    # 資料增強（只用在訓練集）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = CTClassificationDataset("train.csv", image_root, transform=train_transform)
    val_dataset = CTClassificationDataset("val.csv", image_root, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_loss_list = []
    val_loss_list = []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # === 驗證階段 ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)

        print(f"✅ Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # === Early stopping ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task1_resnet18_mac_add_augmentayion.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ 早停於第 {epoch+1} 回合，驗證集 loss 無提升")
                break

    print("🎉 訓練完成，最佳模型儲存為 task1_resnet18_mac.pt")

    # === 評估模型 ===
    model.load_state_dict(torch.load("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task1_resnet18_mac_add_augmentayion.pt"))
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n📊 分類報告：")
    print(classification_report(all_labels, all_preds, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Hemorrhage", "Hemorrhage"], yticklabels=["No Hemorrhage", "Hemorrhage"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/confusion_matrix_task1_add_augmentayion.png")
    plt.show()

    # === 繪製 Loss 曲線（訓練與驗證） ===
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss_list, label="Train Loss", marker='o')
    plt.plot(val_loss_list, label="Val Loss", marker='x')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task1_loss_comparison_add_augmentayion.png")
    plt.show()

if __name__ == "__main__":
    model = train_task1_classification()
