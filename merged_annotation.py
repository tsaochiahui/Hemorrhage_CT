import os
import shutil

# 設定來源與目標資料夾
source_root = "/Users/chia-huitsao/Downloads/PHE-SICH-CT-IDS/SubdatasetC_PNG/annotation/"
target_folder = "/Users/chia-huitsao//Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/merged_annotation"
os.makedirs(target_folder, exist_ok=True)

# 從 annotation1 到 annotation167 搜尋 XML 檔案
for i in range(1, 168):
    folder_name = f"annotation{i}"
    folder_path = os.path.join(source_root, folder_name)
    if not os.path.isdir(folder_path):
        print(f"跳過不存在的資料夾：{folder_path}")
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".xml"):
            src = os.path.join(folder_path, file)
            dst = os.path.join(target_folder, file)
            shutil.copy(src, dst)

print("所有 XML 檔案已複製到 merged_annotation/")