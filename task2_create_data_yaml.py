from pathlib import Path

# 修改成你的完整資料集根路徑
yolo_dataset_path = Path("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/yolo_dataset")

data_yaml = f"""
path: {yolo_dataset_path}
train: images/train
val: images/val
names:
  0: hemorrhage
"""

# 輸出到指定位置
output_path = yolo_dataset_path / "data.yaml"
with open(output_path, "w") as f:
    f.write(data_yaml.strip())

print(f"✅ data.yaml 已產生於：{output_path}")