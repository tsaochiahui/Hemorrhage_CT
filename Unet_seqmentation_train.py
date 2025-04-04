import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import random

# === 自訂 Dataset 類別 ===
class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_ids[idx])
        mask_path = os.path.join(self.mask_dir, self.image_ids[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype('float32')

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1)) / 255.0
        mask = np.expand_dims(mask, 0)
        return image.astype('float32'), mask.astype('float32')

# === 可視化函數 ===
def visualize_prediction(model, dataset, index=0, device="cpu"):
    model.eval()
    image, mask = dataset[index]
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image_tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float().squeeze().cpu().numpy()

    image_np = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
    mask_np = mask.squeeze()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(image_np); plt.title("原始影像"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(mask_np, cmap="gray"); plt.title("真實出血區"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(pred, cmap="gray"); plt.title("模型預測區域"); plt.axis("off")
    plt.show()

# === 評估指標函數 ===
def calculate_metrics(model, dataloader, device="cpu"):
    model.eval()
    TP = TN = FP = FN = 0
    smooth = 1e-6

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

        TP += ((preds == 1) & (masks == 1)).sum().item()
        TN += ((preds == 0) & (masks == 0)).sum().item()
        FP += ((preds == 1) & (masks == 0)).sum().item()
        FN += ((preds == 0) & (masks == 1)).sum().item()

    dice = (2 * TP) / (2 * TP + FP + FN + smooth)
    iou = TP / (TP + FP + FN + smooth)
    sensitivity = TP / (TP + FN + smooth)
    specificity = TN / (TN + FP + smooth)

    print("\n📊 Segmentation Metrics:")
    print(f"Dice Score:     {dice:.4f}")
    print(f"IoU:            {iou:.4f}")
    print(f"Sensitivity:    {sensitivity:.4f}")
    print(f"Specificity:    {specificity:.4f}")

# === 主要程式開始 ===
# 資料路徑（請改成你自己的）
seg_path = "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Task3/seg_dataset"
train_dataset = SegDataset(f"{seg_path}/images/train", f"{seg_path}/masks/train")
val_dataset = SegDataset(f"{seg_path}/images/val", f"{seg_path}/masks/val")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# 模型設定
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
device = torch.device("cpu")
model = model.to(device)

# Loss 函數設定
bce_loss = nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(mode='binary')

def combined_loss(pred, target):
    return bce_loss(pred, target) + dice_loss(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 確保儲存資料夾存在
os.makedirs("weights", exist_ok=True)

# 訓練迴圈
num_epochs = 10
best_loss = float("inf")
loss_history = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = torch.tensor(images).to(device)
        masks = torch.tensor(masks).to(device)

        preds = model(images)
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"✅ Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # 儲存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "weights/unet_task3_best.pt")
        print(f"💾 儲存最佳模型（loss: {best_loss:.4f}）")

    # 每個 epoch 隨機顯示一張預測圖
    random_idx = random.randint(0, len(val_dataset) - 1)
    visualize_prediction(model, val_dataset, index=random_idx, device=device)

# 最終儲存模型
torch.save(model.state_dict(), "weights/unet_task3_final.pt")
print("✅ 訓練完成，模型已儲存為 unet_task3_final.pt")

# 畫出 Loss 曲線
plt.figure()
plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# 評估模型在驗證集的分割指標
calculate_metrics(model, val_loader, device=device)
