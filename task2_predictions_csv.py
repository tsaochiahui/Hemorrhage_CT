import pandas as pd
import os
from pathlib import Path

predictions = []

# 用你的 best.pt 模型
model = YOLO("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/best.pt")

# 圖片資料夾（驗證集）
val_images_dir = Path("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/yolo_dataset/images/val")

# 預測所有 val 圖片
for image_file in sorted(val_images_dir.glob("*.jpg")):
    results = model(str(image_file), conf=0.001)[0]  # 信心低也保留
    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        predictions.append({
            "image": image_file.name,
            "class_id": cls_id,
            "confidence": round(conf, 4),
            "x1": round(x1, 1),
            "y1": round(y1, 1),
            "x2": round(x2, 1),
            "y2": round(y2, 1),
        })

# 輸出 CSV
df = pd.DataFrame(predictions)
output_path = "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task1_val_predictions.csv"
df.to_csv(output_path, index=False)
print(f"✅ 預測結果已儲存：{output_path}")