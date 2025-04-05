import os
import shutil
import random
import xml.etree.ElementTree as ET
from PIL import Image 

# === 原始資料來源 ===
xml_dir = "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/merged_annotation"
image_root = "/Users/chia-huitsao/Downloads/HE-SICH-CT-IDS/SubdatasetB_JPG/set"

# === 目標 YOLO 資料夾結構 ===
base_dir = "/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/yolo_dataset"
for subset in ["train", "val"]:
    os.makedirs(f"{base_dir}/images/{subset}", exist_ok=True)
    os.makedirs(f"{base_dir}/labels/{subset}", exist_ok=True)

# === 抓取 XML 檔名（不含副檔名）===
xml_filenames = [f[:-4] for f in os.listdir(xml_dir) if f.endswith(".xml")]

# === 隨機切分 80% train / 20% val ===
random.seed(42)
random.shuffle(xml_filenames)
split_idx = int(0.8 * len(xml_filenames))
train_list = xml_filenames[:split_idx]
val_list = xml_filenames[split_idx:]

def convert_and_copy(filelist, subset):
    for name in filelist:
        xml_path = os.path.join(xml_dir, name + ".xml")
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 找對應圖片
            img_path = None
            for rootdir, _, files in os.walk(image_root):
                if name + ".jpg" in files:
                    img_path = os.path.join(rootdir, name + ".jpg")
                    break

            if img_path is None:
                print(f"[SKIP] 找不到圖片: {name}.jpg")
                continue

            # 取得圖片大小
            with Image.open(img_path) as im:
                width, height = im.size

            # 轉換成 YOLO 格式
            label_lines = []
            for bbox in root.findall(".//bndbox"):
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                box_w = (xmax - xmin) / width
                box_h = (ymax - ymin) / height

                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

            # 複製圖片與標註
            shutil.copy(img_path, f"{base_dir}/images/{subset}/{name}.jpg")
            with open(f"{base_dir}/labels/{subset}/{name}.txt", "w") as f:
                if label_lines:
                    f.write("\n".join(label_lines))
                # 沒 bbox 就寫空白檔

        except Exception as e:
            print(f"[ERROR] {name}: {e}")

# === 執行轉換與搬移 ===
convert_and_copy(train_list, "train")
convert_and_copy(val_list, "val")

print("✅ 完成！所有圖片與 YOLO 標註已建立（含空白 txt）")