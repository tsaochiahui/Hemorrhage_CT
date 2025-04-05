from ultralytics import YOLO
import pandas as pd
import shutil
import os

# 載入 YOLOv8 的模型（可選：yolov8n.pt, yolov8s.pt, yolov8m.pt...）
model = YOLO("yolov8n.pt")  # n=Nano，最輕量，適合 CPU 訓練

# 訓練模型
results = model.train(
    data="/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/yolo_dataset/data.yaml",  # ← 請確認路徑正確
    epochs=100,
    patience=20,
    imgsz=512,
    batch=4,
    device="cpu",
    project="runs_tensorboard",
    name="task1_yolo",
    verbose=True
)


#事後分析建議    epochs=88,patience=10,