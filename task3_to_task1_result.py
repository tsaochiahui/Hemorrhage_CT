import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Heiti TC'
# === 路徑設定 ===
csv_path = "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task1_classification_labels.csv"                     # 所有圖片的 Ground Truth
val_image_dir = Path("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Task3/seg_dataset/images/val")                  # 驗證集圖片位置
pred_mask_dir = Path("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Task3/seg_dataset/masks/val")                          # segmentation 預測 mask 位置
save_path = "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task3_to_task1_confusion_matrix_val_only.png"       # 混淆矩陣儲存位置


# === 1. 讀入 GT 標註
df_all = pd.read_csv(csv_path)

# === 2. 篩選驗證集的 Ground Truth
val_filenames = [p.name for p in val_image_dir.glob("*.png")]
df_val = df_all[df_all["filename"].isin(val_filenames)].copy()

# === 3. 根據 mask 預測是否有出血
def has_hemorrhage(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)
    return int(np.any(mask_array > 0))  # 有任一非零像素表示有出血

pred_labels = []
for filename in df_val["filename"]:
    mask_file = pred_mask_dir / filename.replace(".jpg", ".png")
    if mask_file.exists():
        pred = has_hemorrhage(mask_file)
    else:
        print(f"⚠️ 找不到預測 mask：{mask_file}")
        pred = 0
    pred_labels.append(pred)

# === 4. 評估分類結果 ===
y_true = df_val["label"].tolist()
y_pred = pred_labels

# 1. 混淆矩陣
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Hemorrhage", "Hemorrhage"])
disp.plot(cmap="Blues")
plt.title("任務三 to 任務一（驗證集）混淆矩陣")
plt.tight_layout()
plt.savefig(save_path)
plt.show()

# 2. 評估指標
print("\n📊任務三模型在驗證集上的分類表現：")
print(classification_report(y_true, y_pred, target_names=["Non-Hemorrhage", "Hemorrhage"], digits=4))