# PHE-SICH-CT-IDS: Hemorrhage CT Scan Dataset

# **題目**

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
| 任務一 | 二元分類任務 | 整張圖：有無出血？ | `ResNet`, `EfficientNet`, `VGG` | 圖像分類架構／不關心位置，只要判斷有沒有 | 資料並無**Non-Hemorrhage** | 補足正常 CT 即可 |
| 任務二 | 出血偵測（Object Detection） | 圖中**哪裡**有出血？（框出位置） | `YOLOv8`, `Faster R-CNN`, `RetinaNet` | 偵測類模型／ 預測「框 + 類別」 |  | 可行 |
| 任務三 | 語意分割（Segmentation） | 圖中**哪些像素**是出血？ | `U-Net`, `U-Net++`, `nnU-Net`, `SegFormer` | 分割類模型／ 預測「每一個像素」的類別 |  | 可行 |
| **任務四** | 多類型出血分類（Multi-label Classification） | 可用於任務一～三 |  |  | 目前僅含一類（basal ganglia SICH） | 需擴充資料 |

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
    your_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    ```
    
2. 將.xml中bndbox轉成yolo格式的.txt
    
    ```python
    label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
    ```
    
    其中<class_id> 目前標注只有hemorrhage（出血）所以<class_id>都是0
    
3. 建立yolov8訓練設定檔data.yaml
    
    ```python
    path: ./  # 專案根目錄
    train: images/train
    val: images/val
    nc: 1  # 類別數（只標出 hemorrhage）
    names: ['hemorrhage']
    ```
    
4. 訓練模型
    1. 使用mac所以指定使用cpu運算
    2. 用的是 YOLOv8 最輕量模型 `yolov8n.pt` +  CPU 訓練
    
    ```python
    from ultralytics import YOLO
    
    model = YOLO("yolov8n.pt")
    
    model.train(
        data="data.yaml",
        epochs=50,           
        imgsz=512,           
        batch=4,            
        device="cpu",        
        workers=0            
    )
    ```
    
5. 測試與視覺化
    1. 單一圖片測試結果
    
    ![image.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/image.png)
    
    ![image.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/image%201.png)
    
    b. 將驗證集的結果輸出成Excel>>>`yolo_val_predictions.xlsx`
    
    其中包含**confidence跟**bounding box 位置
    
    | **image** | **class_id** | **confidence** | **xmin** | **ymin** | **xmax** | **ymax** |
    | --- | --- | --- | --- | --- | --- | --- |
    | 0069_21.jpg | 0 | 0.326057314872742 | 203.434783935547 | 120.868515014648 | 307.211364746094 | 180.443710327148 |
    | 0004_18.jpg | 0 | 0.892451643943787 | 170.176330566406 | 120.84245300293 | 280.908264160156 | 200.291458129883 |
    
    c. 模型準確率結果
    
    | **Class** | **Precision** | **Recall** | **mAP@0.5** | **mAP@0.5:0.95** |
    | --- | --- | --- | --- | --- |
    | hemorrhage | 1 | 0.987094424673276 | 0.994789473684211 | 0.679480727364294 |
    
    d. 混淆矩陣 >>>  hemorrhage／background 
    
    ![confusion_matrix_normalized.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/confusion_matrix_normalized.png)
    
    e. loss curve
    
    ![task2_loss_curve.png](PHE-SICH-CT-IDS%20Hemorrhage%20CT%20Scan%20Dataset%201caa492aa861805391d5fd1e2977b428/task2_loss_curve.png)
    
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
        
    3. 評估指標：Dice score、IoU、Sensitivity、Specificity
        
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
        
        目前模型 **對「背景」很敏感但對「出血」抓得不準**，可能是：
        
        ### 資料不均衡？
        
        - 大部分圖片都是「背景」，出血區太小
        - 👉 可以考慮**加強資料增強（Data Augmentation）**
        
        ### 使用更輕量模型（ResNet34）學不到複雜模式
        
        - 可升級到 `resnet50` 或試試 `U-Net++`
        
        ### 較小的圖片尺寸（如果用的是 256x256）
        
        - 可考慮升級到 `512x512` 看是否改善預測區塊的細節
        </aside>
        
    
8. `coding adjuest（didn’t try）`
    1. **add save Training Loss Curve**
    2. **add Validation Loss Curve(看是否 overfitting）**
    3. add early stopping
    4. add **Dice Loss**
    5. add **BCE Loss**
</aside>
