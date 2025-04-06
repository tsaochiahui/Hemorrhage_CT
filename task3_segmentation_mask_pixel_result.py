import segmentation_models_pytorch as smp
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.rcParams['font.family'] = 'Heiti TC'
# === 1. Dataset 類別定義 ===
class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.mask_paths = sorted(self.mask_dir.glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()
        return image, mask

# === 2. 計算指標與混淆矩陣函數 ===
def calculate_metrics(model, dataloader, device="cpu"):
    model.eval()
    TP = TN = FP = FN = 0
    smooth = 1e-6
    all_preds = []
    all_labels = []

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

        TP += ((preds == 1) & (masks == 1)).sum().item()
        TN += ((preds == 0) & (masks == 0)).sum().item()
        FP += ((preds == 1) & (masks == 0)).sum().item()
        FN += ((preds == 0) & (masks == 1)).sum().item()

        all_preds.extend(preds.view(-1).cpu().numpy())
        all_labels.extend(masks.view(-1).cpu().numpy())

    dice = (2 * TP) / (2 * TP + FP + FN + smooth)
    iou = TP / (TP + FP + FN + smooth)
    sensitivity = TP / (TP + FN + smooth)
    specificity = TN / (TN + FP + smooth)

    print("\n📊 Segmentation Metrics:")
    print(f"Dice Score:     {dice:.4f}")
    print(f"IoU:            {iou:.4f}")
    print(f"Sensitivity:    {sensitivity:.4f}")
    print(f"Specificity:    {specificity:.4f}")

    # === 儲存指標報表 ===
    metrics_df = pd.DataFrame([{
        "Dice": dice, "IoU": iou,
        "Sensitivity": sensitivity, "Specificity": specificity,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN
    }])
    metrics_df.to_csv("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task3_segmentation_metrics.csv", index=False)

    # === 繪製混淆矩陣 ===
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["背景", "出血"])
    disp.plot(cmap="Blues")
    plt.title("Segmentation Confusion Matrix")
    plt.tight_layout()
    plt.savefig("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task3_segmentation_confusion_matrix.png")
    plt.show()

# === 3. 載入資料與模型 ===
seg_path = "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Task3/seg_dataset"
val_dataset = SegDataset(f"{seg_path}/images/val", f"{seg_path}/masks/val")
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
model.load_state_dict(torch.load(
    "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Task3/unet_task3_final.pt",
    map_location="cpu"
))
model.to("cpu")

# === 4. 執行評估 ===
calculate_metrics(model, val_loader, device="cpu")