# PHE-SICH-CT-IDS: Hemorrhage CT Scan Dataset

# **題目**

data source:[https://www.kaggle.com/datasets/naumanalimurad/phe-sich-ct-ids/data](https://www.kaggle.com/datasets/naumanalimurad/phe-sich-ct-ids/data)

**第一階段（需一般高規 GPU 電腦即可）**

**任務一：二元分類（Hemorrhage vs Non-Hemorrhage）**

- 使用 CNN 架構（如 ResNet、EfficientNet）
- 執行資料標準化與增強（如水平翻轉、隨機裁切）
- 評估指標：Accuracy, Precision, Recall, F1-score

**任務二（加分項）：基礎偵測任務**

- 可嘗試以 YOLOv8 或 Faster R-CNN 偵測出血區域（bounding box）
- 視覺化成果並說明模型學習的困難點

---

**第二階段（需要更高算力與深度模型實作能力）**

**任務三：語意分割任務（Segmentation）**

- 模型選擇建議：U-Net、U-Net++、nnU-Net、SegFormer
- 使用出血標註區域的 segmentation mask
- 評估指標：Dice score、IoU、Sensitivity、Specificity

**任務四：多類型出血分類（Multi-label）**

- 分類目標：IVH、IPH、SAH 類型（可多選）
- 可使用多標籤輸出（Sigmoid + BCE Loss）

---

# **題目理解與結論**

|  | 任務內容 | 預測目標 | 使用模型（常見） | 選用理由 |  | 可行性 |
| --- | --- | --- | --- | --- | --- | --- |
| 任務一 | 二元分類任務 | 整張圖：有無出血？ | `ResNet`, `EfficientNet`, `VGG` | 圖像分類架構／不關心位置，只要判斷有沒有 | 資料並無**Non-Hemorrhage標注** | 需反推出標注 |
| 任務二 | 出血偵測（Object Detection） | 圖中**哪裡**有出血？（框出位置） | `YOLOv8`, `Faster R-CNN`, `RetinaNet` | 偵測類模型／ 預測「框 + 類別」 |  | 可行 |
| 任務三 | 語意分割（Segmentation） | 圖中**哪些像素**是出血？ | `U-Net`, `U-Net++`, `nnU-Net`, `SegFormer` | 分割類模型／ 預測「每一個像素」的類別 |  | 可行 |
| **任務四** | 多類型出血分類（Multi-label Classification） | 可用於任務一～三 |  |  | 目前僅含一類（basal ganglia SICH） | 需擴充資料 |

# 檔案說明

work environments: python 3.12.4

<aside>

### task1

| 執行順序 | 檔案 | 產出資料 |  |
| --- | --- | --- | --- |
| 1 | merged_annotation.py | merged_annotation資料夾 |  |
| 2 | output_label_csv.py | task1_classification_labels.csv |  |
| 3 | task1_train_mac_cpu.py | task1_resnet18_mac.pt;task1_loss_comparison.png;confusion_matrix_task1.png |  |
| 4 | task1_train_mac_cpu_add_argmentation.py | task1_resnet18_mac_add_augmentayion.pt;confusion_matrix_task1_add_augmentayion.png;task1_loss_comparison_add_augmentayion.png
 | Data Augmentation |
</aside>

<aside>

### task2

cell test  >>>. `TASK2_V2.ipynb`

| 執行順序 | 檔案 | 產出資料 |  |
| --- | --- | --- | --- |
| 1 | task2_output_yolo_data.py | yolo_dataset資料夾 |  |
| 2 | task2_create_data_yaml.py |  |  |
| 3 | task2_YOLO_train.py | best.pt |  |
| 4 | task2_output_train_result_csv.py | raining_performance_summary.png |  |
| 5 | task2_singlepic_bndbox_show_test.py | yolo_result_0004_18.jpg | skip |
| 6 | task2_predictions_csv.py | task1_val_predictions.csv | skip |
| 7 | task2_predictions_with_none.py | task2_val_predictions_with_none.csv;task2_val_classification_summary.csv;task2_confusion_matrix_summary.png;task2_classification_report.txt | skip |
| 8 | task2_output_report.py | task2_val_predictions_with_none.csv;task2_val_classification_summary.csv;task2_confusion_matrix_summary.png;task2_classification_report.txt |  |
</aside>

<aside>

### task3

cell test  >>>. `task3.ipynb`

| 執行順序 | 檔案 | 產出資料 |  |
| --- | --- | --- | --- |
| 1 | output_seg_dataset.py | seg_dataset資料夾 |  |
| 2 | Unet_seqmentation_train.py | unet_task3_final.pt |  |
| 3 | task3_segmentation_mask_pixel_result.py | task3_segmentation_metrics.csv/task3_segmentation_confusion_matrix.png |  |
| 4 | task3_to_task1_result.py | task3_to_task1_confusion_matrix_val_only.png |  |
</aside>

# 解題思路

## 資料解析

該開源資料 **PHE-SICH-CT-IDS: Hemorrhage CT Scan Dataset**

是由論文《PHE-SICH-CT-IDS: A Benchmark CT Image Dataset for Evaluation Semantic Segmentation, Object Detection and Radiomic Feature Extraction of Perihematomal Edema in Spontaneous Intracerebral Hemorrhage》介紹了一個新的公開數據集，專門用於自發性腦出血（SICH）中的血腫周圍水腫（PHE）的研究。

**資料內容：**

- 120 例 SICH 病患的頭部 CT 掃描，共 7,022 張影像。
- 分為三個子數據集（NIFTI、PNG、JPG 格式）。
- 包含：
    - 語義分割標註（手工標記 PHE）
    - 物件偵測標註（含 PHE 與出血區域）
    - 放射特徵（radiomic features）
    - 患者臨床資料（年齡、性別、是否發生血腫擴張等）

經過分析該資料檔案如下：

提供三種檔案格式（NIFTI、PNG、JPG 格式）

其中 `MedInfo.xlsx` 中有患者資料依據Type Of Hemorrhage其資料分佈如下

|  | count | percentage(%) |
| --- | --- | --- |
| left basal ganglia  | 60 | 50.5 |
| right basal ganglia region | 59 | 49.17 |
| right basal ganglia region、left basal ganglia  | 1 | 0.83 |

---

annotation檔案中標註資料如下：

```python
<?xml version='1.0' encoding='us-ascii'?>
<annotation>
<folder>VOC2007</folder>
<filename>0004_15.jpg</filename>
<path>C:\Users\MiaMia\Desktop\png\set\image4\15.png</path>
<source>
    <database>Unknown</database>
</source>
<size>
    <width>512</width>
    <height>512</height>
    <depth>3</depth>
</size>

<segmented>0</segmented>
    <object>
    <name>a</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
        <xmin>164</xmin>
        <ymin>114</ymin>
        <xmax>285</xmax>
        <ymax>184</ymax>
    </bndbox>
</object>
</annotation>
```

<aside>

- size >>>圖片尺寸>>>512*512*3
- truncated >>>
- difficult >>>是否為難以辨識的物件
- bndbox >>>物件的**邊界框（Bounding Box）**
</aside>

>>>目前提供資料並無提供原始標注訓練集與驗證集

只能使用該檔案提供結果假設此為人工標注之結果進行訓練

## **任務一．**解題思路

<aside>

1. 合併`annotation`中所有`.xml`為**`merged_annotation`**
2. 將所有**`merged_annotation`** 中只要.xml包含`<bndbox>`對應的圖片視為**Hemorrhage(1)**，其他標注成**Non-Hemorrhage(0)**產出`task1_classification_labels.csv`，該檔案當作人工標注結果
3. 訓練模型
    1. 使用`ResNet` （因為是用mac cpu晶片跑,理論上`EfficientNet`效果較佳）
        - 因為`ResNet` 是為了解決較深神經網路存在的，`EfficientNet` 考量深度寬度解析度理論上模型更高效
    2. 訓練結果
        1. 評估指標：Accuracy, Precision, Recall, F1-score
            
            ```python
            📊 分類報告：
                          precision    recall  f1-score   support
            
                       0     0.9909    0.9873    0.9891       552
                       1     0.9521    0.9653    0.9586       144
            
                accuracy                         0.9828       696
               macro avg     0.9715    0.9763    0.9739       696
            weighted avg     0.9829    0.9828    0.9828       696
            ```
            
            | 指標 | 說明 |
            | --- | --- |
            | accuracy | 整體準確率98.28%(總共696張，預測對了684張) |
            | macro avg | 每個類別指標的「平均值」，不考慮類別比例。適合看模型是否偏袒某類。 |
            | weighted avg | 加權平均：考慮 support（樣本數），接近整體平均表現。 |
        2. 混淆矩陣
            
            ![confusion_matrix_task1.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/confusion_matrix_task1.png)
            
            |  | 預測為無出血 | 預測為有出血 |
            | --- | --- | --- |
            | 實際無出血 | 545（TN） | 7（FP） |
            | 實際有出血 | 545（TN） | 139（TP） |
            - **準確率非常高**（總共 696 張，錯 12 張）
            - 假陽性（False Positive）= 7 張
            - 假陰性（False Negative）= 5 張 → 對醫療來說這是特別關鍵的（不能漏診）
        3. loss curve
            
            ![task1_loss_comparison.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/task1_loss_comparison.png)
            
            ### 📉 訓練 Loss（藍色實線）：
            
            - 持續下降 → 模型確實有學到
            
            ### 📈 驗證 Loss（橘色虛線）：
            
            - 前幾輪穩定 → 到了第 6 輪**突然上升**，代表：
                - 模型開始過擬合（Overfitting）
                - **Early Stopping 機制**有成功啟動（應該停在第 6 輪）
            - 總結>>>加強資料增強（Data Augmentation），或使用更輕量模型觀察
    3. Add Data Augmentation  訓練結果
        1. 評估指標：Accuracy, Precision, Recall, F1-score
            
            ```python
            📊 分類報告：
                          precision    recall  f1-score   support
            
                       0     0.9891    0.9873    0.9882       552
                       1     0.9517    0.9583    0.9550       144
            
                accuracy                         0.9813       696
               macro avg     0.9704    0.9728    0.9716       696
            weighted avg     0.9814    0.9813    0.9813       696
            ```
            
        2. 混淆矩陣
            
            ![confusion_matrix_task1_add_augmentayion.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/confusion_matrix_task1_add_augmentayion.png)
            
        3. loss curve
            
            ![task1_loss_comparison_add_augmentayion.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/task1_loss_comparison_add_augmentayion.png)
            
4. 結論
    1. 整體變化很小（可視為效果持平）：
        
        
        | 指標 | 增強前 | 增強後 |
        | --- | --- | --- |
        | accuracy | 98.28% | 98.13% |
        | hemorrhage Recall | **0.9653** | **0.9583** ↓ 一些 |
        | hemorrhage F1-score | 0.9586 | 0.9550 |
        
        > 結論：增強後模型在泛化能力上沒有下降，但也沒有明顯提升
        > 
    2.  為什麼 Data Augmentation 看起來沒明顯提升？
        - 原始資料質量已經很好（分類本身不太難）
        - 加的增強偏「輕量」，不會改變太多資訊
        - 資料集夠大時，增強效益可能不明顯
    3. solution
        
        
        | 做法 | 原因與目的 |
        | --- | --- |
        | ➕ 測試集預測 | 更公平檢查模型泛化能力（不是 val set） |
        | 📈 繪製 ROC Curve/AUC | 評估模型在「不同判斷閾值」下表現，尤其針對出血很重要 |
        | 🧪 單張推論 | 方便模型部署（給 CT 圖片 → 預測結果） |
        | 🔍 Grad-CAM | 看模型是依據圖上哪裡「做出判斷」 → 提升可信度 |
</aside>

## **任務二．**解題思路

<aside>

1. 將檔案中圖面分為訓練集與驗證集檔案格式如下：
    
    ```python
    yolo_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    ```
    
    1. 所有**`merged_annotation`** 中只要.xml包含`<bndbox>`對應的圖片視為**Hemorrhage(1)**，其他標注成**Non-Hemorrhage(0)**產出空白`.txt`
    2. 其中.txt格式如下：
    
    ```python
    <類別> <x_center> <y_center> <width> <height>
    ```
    
    - `x_center`, `y_center`, `width`, `height` 都是相對於圖片尺寸（0~1）
    - **`0` 是類別（只有一類 hemorrhage）**
    
    <aside>
    
    013_00~31.xml命名錯誤無法找到相對應圖片（應該是0013_00~31.xml）
    
    </aside>
    
2. 產生data.yaml
3. 建立yolov8訓練設定檔data.yaml
    
    ```python
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
    ```
    
4. 訓練模型
    1. 使用mac所以指定使用cpu運算
    2. 用的是 YOLOv8 最輕量模型 `yolov8n.pt` +  CPU 訓練
    
    ```python
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
    ```
    
    <aside>
    
    設定early stopping >>>>依據事後分析收斂在epochs=69,
    
    建議參數改成    epochs=88,patience=10,
    
    </aside>
    
    c.  訓練結果分析
    
    ![training_performance_summary.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/training_performance_summary.png)
    
      
    
    - raining Loss Curve
        - **🔵 Box Loss**：框的位置預測誤差（bounding box 的準確度）
        - **🟠 Class Loss**：出血 vs 非出血 的分類誤差
    - **mAP Performance Curve**
        - **🔵 mAP@0.5**：預測框與標註框重疊超過 50% 就算正確 → 最常見的精準度評估指標
        - **🟠 mAP@0.5:0.95**：不同 IoU 門檻（0.5～0.95）下的平均值 → 更嚴格更全面的指標
            
            <aside>
            
            - mAP@0.5 很快就達到 **0.99+** → 框的位置預測非常準確
            - mAP@0.5:0.95 雖然較低（最高約 0.71），但仍穩定上升 → 模型逐漸學會處理更困難的案例
            - `綠線` 標記 **最佳 mAP@0.5 所在 epoch（第 69 回合）** → 你可以回頭分析那個模型的表現
            - `紅線` 為你訓練設定的上限（第 100 回合）→ 模型持續進步，所以沒有早停
            </aside>
            
    - 結論
        
        
        | 項目 | 結果說明 |
        | --- | --- |
        | 模型有學習嗎？ |  Loss 持續下降，mAP 持續上升 |
        | 模型準確嗎？ | mAP@0.5 ≈ 0.99，代表預測框與實際非常接近 |
        | 有過擬合嗎？ | 無，loss 沒有反彈，mAP 沒下降 |
        | Early Stopping | 未觸發（模型一直有進步） |
        | 可使用 epoch | 建議用 epoch=69 的模型作為最佳模型（`best.pt`） |
5. 測試與視覺化
    1. 單一圖片測試結果
    
    ![image.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/image.png)
    
    b. 將驗證集的結果輸出成Excel>>>`tas2_val_predictions_with_none.csv`
    
    | **image** | **class_id** | **confidence** | **x1** | **y1** | **x2** | **y2** | **class_name** |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | **0001_07.jpg** | none | 0.0 |  |  |  |  | Non-Hemorrhage |
    | **0001_09.jpg** | none | 0.0 |  |  |  |  | Non-Hemorrhage |
    | **0001_11.jpg** | none | 0.0 |  |  |  |  | Non-Hemorrhage |
    | **0001_12.jpg** | none | 0.0 |  |  |  |  | Non-Hemorrhage |
    | **0001_15.jpg** | 0 | 0.8223 | 230.5 | 164.7 | 329.5 | 247.5 | Hemorrhage |
    | **0001_17.jpg** | 0 | 0.7918 | 237.1 | 178.1 | 327.6 | 246.9 | Hemorrhage |
    
    <aside>
    
    - **<class_id> =** none →Non-Hemorrhage
    - **<class_id> = 0** →Hemorrhage
    </aside>
    
    c. 模型準確率結果輸出成Excel>>>`task2_val_classification_summary.csv`
    
    | **image** | **ground_truth** | **predicted** | **classification** |
    | --- | --- | --- | --- |
    | **0001_07.jpg** | Non-Hemorrhage | Non-Hemorrhage | TN |
    | **0001_09.jpg** | Non-Hemorrhage | Non-Hemorrhage | TN |
    | **0001_11.jpg** | Non-Hemorrhage | Non-Hemorrhage | TN |
    | **0001_12.jpg** | Hemorrhage | Non-Hemorrhage | FN |
    | **0001_15.jpg** | Hemorrhage | Hemorrhage | TP |
    
    d. 混淆矩陣
    
    ![task2_confusion_matrix_summary.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/task2_confusion_matrix_summary.png)
    
    <aside>
    
    |  | **Pred: Hemorrhage** | **Pred: Non-Hemorrhage** |
    | --- | --- | --- |
    | **GT: Hemorrhage** | 140 ✅ (TP) | 1 ❌ (FN) |
    | **GT: Non-Hemorrhage** | 17 ❌ (FP) | 537 ✅ (TN) |
    - **TP（真正例）140 張**：模型正確預測出血。
    - **FN（假負例）1 張**：模型漏判出血。
    - **FP（假正例）17 張**：模型誤判無出血為出血。
    - **TN（真負例）537 張**：模型正確判定無出血。
    
    ```python
    
                    precision    recall  f1-score   support
    
        Hemorrhage     0.8917    0.9929    0.9396       141
    Non-Hemorrhage     0.9981    0.9693    0.9835       554
    
          accuracy                         0.9741       695
         macro avg     0.9449    0.9811    0.9616       695
      weighted avg     0.9766    0.9741    0.9746       695
    ```
    
    | 指標 | Hemorrhage | Non-Hemorrhage | 整體（Accuracy） |
    | --- | --- | --- | --- |
    | **Precision** | 0.8917 | 0.9981 | 模型預測為出血的結果有 89.17% 是對的。 |
    | **Recall** | 0.9929 | 0.9693 | 模型抓到了 99.29% 的真實出血圖片，只有 1 張漏掉。 |
    | **F1-score** | 0.9396 | 0.9835 | 出血類別整體表現平衡且優秀。 |
    | **Accuracy** | - | - | **97.41%**，整體預測正確率很高。 |
    - 結論
        - **出血類別的 Recall 非常高（0.9929）** → 幾乎沒有漏判出血，非常適合臨床使用。
        - **Non-Hemorrhage 類別的 Precision 非常高（0.9981）** → 幾乎不會誤判健康影像為出血。
        - **稍微有點 False Positive（17 張）** → 可以考慮微調模型閾值或增加負樣本以改善。
        - **整體 Accuracy 達到 97.4%，F1-score 平衡度也很高。**
    - 改善
        - 若任務的目的是「**不要漏判出血**」，那這樣的 Recall 非常合適。
        - 若你要進一步壓低 **誤報率（FP）**，可考慮：
            - 提高置信閾值 (`conf` 設為 0.3 ~ 0.5)
            - 加強 Non-Hemorrhage 樣本訓練
            - 或者使用 ensemble 模型提高穩定度
    </aside>
    
</aside>

## **任務三．**解題思路

<aside>

1. 模型選用
    1. Apple M3（ARM 架構）>>>>選用U-Net
2. 將檔案中圖面分為訓練集與驗證集檔案格式如下：
    
    ```python
    seg_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── masks/
    │   ├── train/
    │   └── val/
    ```
    
    <aside>
    
    不同於任務二用annotation中的bounding box提供label，任務三是標出「每個像素」是否為出血，所以label 使用檔案中已經label 好的jpg檔
    
    </aside>
    
3. 建立 PyTorch Dataset 與 Dataloader
4. 建立 U-Net 模型
5. 設定損失與訓練迴圈
6. 訓練模型並驗證
7. 結果
    1.  **Loss Curve**
        
        ![output.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/output.png)
        
    2. 影像結果輸出
        
        ![Snipaste_2025-04-04_15-00-36.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/Snipaste_2025-04-04_15-00-36.png)
        
    3. 混淆矩陣
        
        ![task3_segmentation_confusion_matrix.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/task3_segmentation_confusion_matrix.png)
        
        ### 🔍 混淆矩陣數據解析：
        
        |  | 預測為背景 | 預測為出血 |
        | --- | --- | --- |
        | **實際是背景** | `1.8e+08` ✅ | `95,282` ❌ |
        | **實際是出血** | `28,097` ❌ | `40,000` ✅ |
        
        | 名稱 | 定義 | 解釋 |
        | --- | --- | --- |
        | **True Negative (TN)** | `背景→背景` | 模型正確預測背景（**1.8 億**個像素）✅ |
        | **False Positive (FP)** | `背景→出血` | 模型誤把背景當成出血（9 萬像素）❌ |
        | **False Negative (FN)** | `出血→背景` | 模型漏掉出血（2.8 萬像素）❌ |
        | **True Positive (TP)** | `出血→出血` | 模型正確預測出血（4 萬像素）✅ |
        
        ### 📊 初步推論：
        
        - **背景量遠大於出血**：這在醫學 segmentation 任務中常見，屬於嚴重類別不平衡。
        - **模型容易漏掉出血（FN 較高）**：代表有出血但預測為背景的像素不低。
        - **假陽性（FP）也存在**：模型誤將正常區域預測為出血。
    4. 評估指標：Dice score、IoU、Sensitivity、Specificity
        
        ```python
        📊 Segmentation Metrics:
        Dice Score:     0.3936
        IoU:            0.2450
        Sensitivity:    0.5876
        Specificity:    0.9995
        ```
        
        1. Dice Score: **0.3936 → 表示模型預測與 Ground Truth 重疊不到 40%**
            - 這還偏低，代表模型還沒很準確地學會抓出出血區域
        2. Intersection over Union = 0.2450**→ 預測與標註的重疊比例僅約 24.5%**
            - 通常語意分割任務 IoU > 0.5 才被認為是「可用」水準
            - 所以目前結果表示模型預測位置偏差還滿大的
        3. **Sensitivity（Recall）= 0.5876**
            - 表示：**所有有出血的像素中，有多少被模型成功抓出來**
            - 公式：`TP / (TP + FN)`
            
             **58.76% → 還 OK，但有約 41% 出血像素沒被模型抓出來**
            
        4. **Specificity = 0.9995**
            - 表示：**背景區域中，有多少被模型正確辨識為背景**
            - 公式：`TN / (TN + FP)`
            
            **99.95% → 非常高，代表模型幾乎不會誤判背景為出血**
            
    
    <aside>
    
    ### 🚩segmentation mask 方式驗證NonHemorrhage/Hemorrhage
    
    1. 混淆矩陣
        
        ![task3_to_task1_confusion_matrix_val_only.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/task3_to_task1_confusion_matrix_val_only.png)
        
    2. 評估指標：Accuracy, Precision, Recall, F1-score
    
    ```python
    
    📊任務三模型在驗證集上的分類表現：
                    precision    recall  f1-score   support
    
    Non-Hemorrhage     0.9717    1.0000    0.9857       550
        Hemorrhage     1.0000    0.8881    0.9407       143
    
          accuracy                         0.9769       693
         macro avg     0.9859    0.9441    0.9632       693
      weighted avg     0.9776    0.9769    0.9764       693
    ```
    
    </aside>
    
- `coding adjuest（didn’t try）`
    
    
    1. **add save Training Loss Curve**
    2. **add Validation Loss Curve(看是否 overfitting）**
    3. add early stopping
    4. add **Dice Loss**
    5. add **BCE Loss（no ground true）**
    6. add **Data Augmentation. (didn’t write)**
    7. 預測區域跟原始圖片疊加**. (didn’t write)**
    8. 改成GPU(”mps”)運算
</aside>

## **任務四．**(無labels資料無法實作)

- 
    
    [note](https://www.notion.so/note-1cca492aa86180228d96ef3f637573ad?pvs=21)
    

<aside>

## 結構總覽

| 步驟 | 說明 |
| --- | --- |
| 1️⃣ | 數據格式範例（CSV） |
| 2️⃣ | `Dataset` 類別設計（多標籤） |
| 3️⃣ | 模型架構（ResNet + Sigmoid 輸出） |
| 4️⃣ | 損失函數：`BCEWithLogitsLoss` |
| 5️⃣ | 預測 & 評估指標（micro/macro F1-score） |

<aside>

---

## 1️⃣ 假設資料格式：`task4_labels.csv`

```
csv
複製編輯
image,IVH,IPH,SAH
ct1.png,1,0,1
ct2.png,0,1,0
ct3.png,0,0,0
```

---

## 2️⃣ Dataset 類別（多標籤）

```python
python
複製編輯
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import torch

class MultiLabelCTDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_root / row["image"]
        image = Image.open(image_path).convert("RGB")
        label = torch.tensor([row["IVH"], row["IPH"], row["SAH"]], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label
```

---

## 3️⃣ 模型架構 + Sigmoid 輸出

```python
python
複製編輯
import torch.nn as nn
from torchvision import models

class MultiLabelResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 3)  # 三類輸出

    def forward(self, x):
        return self.backbone(x)  # 不加 Sigmoid，交給 loss 自動處理
```

---

## 4️⃣ 損失函數 + Optimizer

```python
python
複製編輯
model = MultiLabelResNet()
criterion = nn.BCEWithLogitsLoss()  # 多標籤用這個
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

```

---

## 5️⃣ 預測 & 評估

```python
python
複製編輯
from sklearn.metrics import classification_report

# 預測時：
outputs = model(images)  # shape: [B, 3]
probs = torch.sigmoid(outputs)
preds = (probs > 0.5).int()  # shape: [B, 3]

# labels: [B, 3]
print(classification_report(labels.cpu().numpy(), preds.cpu().numpy(), target_names=["IVH", "IPH", "SAH"]))

```

---

## ✅ 補充建議：

- **輸出 Sigmoid** 適合用於「可以同時出現多類」的任務
- **BCE Loss** 是最適合處理這種多標籤情境的損失函數
- 可以用 `F1-micro`（整體）或 `F1-macro`（每類平均）觀察分類效果
</aside>

<aside>

## 可能結果模擬

### 多標籤分類：模型輸出 logits → sigmoid 機率 → 損失計算（BCE Loss）

假設我們有 2 張 CT 影像作為輸入，目標是預測 3 類出血：IVH、IPH、SAH

→ 就是一個 **`[batch_size=2, num_classes=3]` 的矩陣**

### 🧾 1️⃣ 模型輸出 logits（原始分數）：

|  | IVH | IPH | SAH |
| --- | --- | --- | --- |
| 圖1 | 2.0 | -1.2 | 0.5 |
| 圖2 | -0.5 | 0.8 | 1.5 |

---

### 🔁 2️⃣ 經過 **Sigmoid** 轉換成機率（每個欄位跑一次 sigmoid）：

|  | IVH | IPH | SAH |
| --- | --- | --- | --- |
| 圖1 | 0.88 | 0.23 | 0.62 |
| 圖2 | 0.38 | 0.69 | 0.82 |

---

### ✅ 3️⃣ Ground Truth（實際標籤）：

|  | IVH | IPH | SAH |
| --- | --- | --- | --- |
| 圖1 | 1 | 0 | 1 |
| 圖2 | 0 | 1 | 1 |

---

### 📉 4️⃣ BCE Loss（每個位置計算損失）：

BCE loss 計算公式為：

BCE(p,y)=−[y⋅log⁡(p)+(1−y)⋅log⁡(1−p)]

BCE(p,y)=−[y⋅log(p)+(1−y)⋅log(1−p)]

套用到每個位置，得到如下損失矩陣（可用 `.mean()` 或 `.sum()` 做總 loss）：

|  | IVH | IPH | SAH |
| --- | --- | --- | --- |
| 圖1 | 0.13 | 0.26 | 0.47 |
| 圖2 | 0.47 | 0.37 | 0.20 |

### 5️⃣你的輸入資料回顧（模型輸出 + 預測機率 + Ground Truth）：

| 圖片 | 類別 | Logit | 機率（sigmoid） | Ground Truth | BCE Loss | 解釋 |
| --- | --- | --- | --- | --- | --- | --- |
| 圖1 | IVH | 2.0 | 0.88 | 1 | 0.13 | 預測很好 ✅ |
| 圖1 | IPH | -1.2 | 0.23 | 0 | 0.26 | 機率低，正確 ✅ |
| 圖1 | SAH | 0.5 | 0.62 | 1 | 0.47 | 有點偏，Loss較大 ❌ |
| 圖2 | IVH | -0.5 | 0.38 | 0 | 0.47 | 有點高了，還OK 😐 |
| 圖2 | IPH | 0.8 | 0.69 | 1 | 0.37 | 有點低了，還OK 😐 |
| 圖2 | SAH | 1.5 | 0.82 | 1 | 0.20 | 預測不錯 ✅ |

### 6️⃣ 結論

|  | 解釋 |
| --- | --- |
| 圖1 | 確定是IVH / 確定不是IPH / 可能是SAH |
| 圖2 | 也許不是IVH / 也許是IPH /確定是SAH |
- **機率表示模式預測是或不是的機率;loss表示個機率的可信度**
</aside>

## 假設我這種多標籤模型有一類特別不準要怎麼處理

<aside>

### 😕 問題可能出在：

| 問題類型 | 可能原因 |
| --- | --- |
| 類別不準（Recall 很低） | 資料太少、特徵不明顯、模型忽略它 |
| 預測錯誤很多（Precision 很低） | 類別混淆度高、模型常誤判其他類為它 |

---

## ✅ 解法總整理（推薦從上往下試）：

---

### 1️⃣ 🔁 **資料層面：重新檢查類別分布**

```python
python
複製編輯
df["label"].value_counts()
```

- 有可能那一類（例如 SAH）樣本數太少 → 導致模型學不到
- 解法：
    - **資料平衡**：嘗試讓 SAH 的樣本增加（可以使用 oversampling）
    - **資料增強**：針對該類別進行較多的圖像增強（flip, rotate, contrast）

---

### 2️⃣ ⚖️ **Loss 權重調整（class weighting）**

讓 loss 更重視預測差的類別！

```python
python
複製編輯
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 1.0, 2.0]))
```

↑ 若第 3 類（SAH）太弱，將權重設為 2.0

這樣該類錯誤時，loss 會變大 → 模型會「更努力學」那一類！

---

### 3️⃣ 🧠 **觀察混淆類別 → 加強區分**

- 是否 IPH / SAH 類型混淆？
- 如果是，可以：
    - 加入 **注意力機制（Attention）**
    - 加強圖像對比度、邊緣清晰度
    - 用「焦點引導增強」（例如只增強腦室區）

---

### 4️⃣ 📊 **分開評估各類指標，監控訓練進度**

訓練中 plot 每類的：

- Precision / Recall / F1-score 走勢
- 看是否只有某一類學不起來，或是過擬合

---

### 5️⃣ 💡 **多任務學習（進階）**

> 如果某類學不好，甚至可以試著用多任務學習
> 
> 
> 把它當作獨立的輸出分支來訓練 → 有時能提升泛化力
> 

---

## 🎯 一句總結：

> 多標籤學習中，如果某類特別不準，你可以試著「給它更多注意力」—— 不論是透過資料、loss、還是特徵提取。
> 
</aside>

</aside>
