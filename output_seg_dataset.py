import os
import shutil
import random
from pathlib import Path

# 資料夾設定
root_dir = Path("/Users/chia-huitsao/Downloads/PHE-SICH-CT-IDS/SubdatasetC_PNG")  # 改成實際路徑
image_root = root_dir / "set"
mask_root = root_dir / "label"
output_base = Path("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Task3/seg_dataset")
(image_train := output_base / "images/train").mkdir(parents=True, exist_ok=True)
(image_val := output_base / "images/val").mkdir(parents=True, exist_ok=True)
(mask_train := output_base / "masks/train").mkdir(parents=True, exist_ok=True)
(mask_val := output_base / "masks/val").mkdir(parents=True, exist_ok=True)

# 收集成對影像與 mask
paired = []
for mask_dir in mask_root.glob("label*/"):
    if not mask_dir.is_dir():
        continue
    label_id = mask_dir.name.replace("label", "")
    image_dir = image_root / f"image{label_id}"
    if not image_dir.exists():
        continue
    for mask_path in mask_dir.glob("*.png"):
        image_path = image_dir / mask_path.name
        if image_path.exists():
            paired.append((image_path, mask_path))

# 分割資料集（80% 訓練 / 20% 驗證）
random.seed(42)
random.shuffle(paired)
split = int(0.8 * len(paired))
train_samples = paired[:split]
val_samples = paired[split:]

# 複製檔案
def copy(pairs, img_out, mask_out):
    for img_path, mask_path in pairs:
        shutil.copy(img_path, img_out / img_path.name)
        shutil.copy(mask_path, mask_out / mask_path.name)

copy(train_samples, image_train, mask_train)
copy(val_samples, image_val, mask_val)

print(f"整理完成，共有影像 {len(paired)} 張，其中訓練 {len(train_samples)}，驗證 {len(val_samples)}")