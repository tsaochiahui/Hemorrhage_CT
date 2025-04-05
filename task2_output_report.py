from ultralytics import YOLO
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# === Step 1: 預測驗證圖片 ===
model = YOLO("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/best.pt")  # ← 換成你的模型路徑
val_images_dir = Path("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/yolo_dataset/images/val")
label_dir = Path("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/yolo_dataset/labels/val")

predictions = []
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
        predictions.append({
            "image": image_file.name,
            "class_id": "none",
            "confidence": 0.0,
            "x1": None,
            "y1": None,
            "x2": None,
            "y2": None,
        })

pred_df = pd.DataFrame(predictions)
pred_df["class_name"] = pred_df["class_id"].apply(lambda x: "Hemorrhage" if str(x) != "none" else "Non-Hemorrhage")
pred_df.to_csv("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task2_val_predictions_with_none.csv", index=False)
print("✅ 已儲存預測結果：task2_val_predictions_with_none.csv")

# === Step 2: 比對 Ground Truth 並分類 TP / FP / FN / TN ===
gt_dict = {}
for txt_file in sorted(label_dir.glob("*.txt")):
    with open(txt_file) as f:
        lines = f.readlines()
    gt_dict[txt_file.stem + ".jpg"] = 1 if lines else 0

records = []
pred_grouped = pred_df.groupby("image")

for image_name, group in pred_grouped:
    has_gt = gt_dict.get(image_name, 0)
    has_pred = (group["class_id"] != "none").any()

    if has_pred and has_gt:
        result = "TP"
    elif has_pred and not has_gt:
        result = "FP"
    elif not has_pred and has_gt:
        result = "FN"
    else:
        result = "TN"

    records.append({
        "image": image_name,
        "ground_truth": "Hemorrhage" if has_gt else "Non-Hemorrhage",
        "predicted": "Hemorrhage" if has_pred else "Non-Hemorrhage",
        "classification": result
    })

summary_df = pd.DataFrame(records)
summary_df.to_csv("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task2_val_classification_summary.csv", index=False)
print("✅ 已儲存分類對照表：task2_val_classification_summary.csv")

# === Step 3: 混淆矩陣 + 分類報告 ===
y_true = summary_df["ground_truth"]
y_pred = summary_df["predicted"]

# 混淆矩陣圖
cm = confusion_matrix(y_true, y_pred, labels=["Hemorrhage", "Non-Hemorrhage"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Hemorrhage", "Non-Hemorrhage"],
            yticklabels=["Hemorrhage", "Non-Hemorrhage"])
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.title("任務一 - 混淆矩陣")
plt.tight_layout()
plt.savefig("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task2_confusion_matrix_summary.png")
plt.show()

# 分類報告（precision, recall, f1-score, accuracy）
print("\n📊 分類報告（Classification Report）：")
report = classification_report(y_true, y_pred, digits=4, output_dict=False)
print(report)

# 若你也想儲存成 txt：
with open("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task2_classification_report.txt", "w") as f:
    f.write(classification_report(y_true, y_pred, digits=4))