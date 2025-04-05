import pandas as pd
import os
from pathlib import Path
from ultralytics import YOLO

predictions = []

model = YOLO("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/best.pt")
val_images_dir = Path("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/yolo_dataset/images/val")

for image_file in sorted(val_images_dir.glob("*.jpg")):
    results = model(str(image_file), conf=0.001)[0]
    boxes = results.boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
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
    else:
        # 沒預測框 → 預測為「non-Hemorrhage」
        predictions.append({
            "image": image_file.name,
            "class_id": "none",        # ← 表示無出血
            "confidence": 0.0,
            "x1": None,
            "y1": None,
            "x2": None,
            "y2": None,
        })

# 輸出 CSV
df = pd.DataFrame(predictions)
output_path = "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/tas2_val_predictions_with_none.csv"
df.to_csv(output_path, index=False)
print(f"✅ 預測結果（含 non-Hemorrhage）已儲存：{output_path}")