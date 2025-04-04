import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
        folder = f"image{int(row['filename'][:4])}"   # image123
        img_path = self.image_root / folder / row["filename"]
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

# === 2. 訓練流程（CPU優化版） ===
def train_task1_classification():
    # ✅ 設定檔案與路徑
    csv_path = "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task1_classification_labels.csv"
    image_root = "/Users/chia-huitsao/Downloads/PHE-SICH-CT-IDS/SubdatasetC_PNG/set"

    # ✅ CPU 訓練參數（較小模型與 batch）
    batch_size = 4
    num_epochs = 10
    device = "cpu"

    # 分割 train / val 集
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)

    # 影像處理（簡單 Resize + Tensor）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 建立 Dataset 與 Dataloader
    train_dataset = CTClassificationDataset("train.csv", image_root, transform)
    val_dataset = CTClassificationDataset("val.csv", image_root, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    # ✅ 使用 ResNet18 做分類
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 二元分類
    model = model.to(device)

        # ✅ 損失與優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # ✅ 開始訓練
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

        print(f"✅ Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader):.4f}")

    # ✅ 儲存模型
    torch.save(model.state_dict(), "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task1_resnet18_mac.pt")
    print("🎉 訓練完成，模型已儲存為 task1_resnet18_mac.pt")

