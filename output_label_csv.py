import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

# 1. 設定路徑
image_root = Path("/Users/chia-huitsao/Downloads/PHE-SICH-CT-IDS/SubdatasetC_PNG/set")
xml_dir = Path("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/merged_annotation")

# 2. 收集標註資料
records = []

for xml_file in sorted(xml_dir.glob("*.xml")):
    filename = xml_file.stem + ".png"         # e.g. 0001_00.png

    # 嘗試取得正確的 folder name（跳過不符合的）
    try:
        prefix = filename[:4]  # 嘗試解析前四碼
        prefix_num = str(int(prefix))  # 去除前導 0，例如 '0112' → 112
        folder_name = f"image{prefix_num}"
    except ValueError:
        print(f"⚠️ 無法解析前綴碼：{filename}")
        continue

    image_path = image_root / folder_name / filename

    if not image_path.exists():
        print(f"⚠️ 找不到圖片：{image_path}")
        continue

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        has_hemorrhage = 1 if root.find("object") is not None else 0
        records.append({"filename": filename, "label": has_hemorrhage})
    except ET.ParseError:
        print(f"❌ XML 解析錯誤：{xml_file.name}")
    except Exception as e:
        print(f"❌ 錯誤處理 XML：{e} ({xml_file.name})")

# 3. 輸出成 CSV
df = pd.DataFrame(records)
csv_path = Path("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/task1_classification_labels.csv")
df.to_csv(csv_path, index=False)

print(f"✅ 已產出 CSV：{csv_path}，共 {len(df)} 筆資料")