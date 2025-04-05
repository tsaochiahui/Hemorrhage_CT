from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 載入訓練好的模型（記得改成你自己訓練出的 best.pt 路徑）
model = YOLO("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/best.pt")

# 載入你要推論的圖片
img_path = "/Users/chia-huitsao/Downloads/HE-SICH-CT-IDS/yolo_dataset/images/val/0004_18.jpg"  # <- 改成你的圖片路徑
results = model(img_path)

# 顯示 bounding box 結果
res_plotted = results[0].plot()  # 這會畫上 bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(res_plotted)
plt.axis("off")
plt.title("推論結果 - YOLOv8 出血區域") 
plt.show()
cv2.imwrite("/Users/chia-huitsao/Documents/PHE-SICH-CT-IDS/Hemorrhage_CT/yolo_result_0004_18.jpg", res_plotted)
print("圖片已儲存為 yolo_result_0004_18.jpg")