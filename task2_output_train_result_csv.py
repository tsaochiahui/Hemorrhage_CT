import pandas as pd
import matplotlib.pyplot as plt

# 修改為你實際的訓練輸出資料夾
log_path = "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/results.csv"

# 讀取訓練紀錄
df = pd.read_csv(log_path)

# 找最佳 mAP epoch
best_map_epoch = df["metrics/mAP50(B)"].idxmax()
best_map = df.loc[best_map_epoch, "metrics/mAP50(B)"]
early_stop_epoch = df["epoch"].max() + 1

# 畫圖
plt.figure(figsize=(12, 6))

# Loss 曲線
plt.subplot(1, 2, 1)
plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss", marker="o")
plt.plot(df["epoch"], df["train/cls_loss"], label="Class Loss", marker="x")
plt.axvline(early_stop_epoch, color="red", linestyle="--", label="Early Stop")
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# mAP 曲線
plt.subplot(1, 2, 2)
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5", marker="o")
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", marker="x")
plt.axvline(best_map_epoch, color="green", linestyle="--", label=f"Best mAP Epoch: {best_map_epoch}")
plt.axvline(early_stop_epoch, color="red", linestyle="--", label="Early Stop")
plt.title("mAP Performance Curve")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/training_performance_summary.png")
plt.show()

print(f"✅ Best mAP@0.5: {best_map:.4f} at epoch {best_map_epoch}")
print(f"⏹️ Early Stopping at epoch {early_stop_epoch}")