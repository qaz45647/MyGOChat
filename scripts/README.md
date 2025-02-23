# 說明文件


📢 部分說明文件是由 AI 協助撰寫的，雖經過人工檢查和調整，但仍可能存在錯誤，敬請見諒。

📢請注意，requirements.txt 中不包含所有程式所需的套件，請自行安裝。

## Image_Tagging_Application.py

**圖片標註器**：本工具是一個 GUI 應用程式，主要用於對圖片的情境進行標籤分類。

![image.png](https://truth.bahamut.com.tw/s01/202502/c12f57b2fa3b458b84275270f2041f39.JPG)

**主要功能包括**：

- **圖片載入**：自動讀取 `IMG` 資料夾內的圖片。
- **標籤選擇**：
    - 提供四大類標籤：情緒、態度、互動、語調。
    - 可透過按鈕選擇或取消標籤。
- **圖片瀏覽**：
    - 可透過「上一張」、「下一張」按鈕切換圖片。
    - 支援輸入圖片編號直接跳轉。
- **結果儲存**：
    - 標記結果將存入 `result.csv`。
    - 可讀取先前標記結果，避免重複標記。

**輸入格式**：

- 程式會讀取同目錄下的 `IMG` 資料夾內的圖片，支援格式：`jpg`, `jpeg`, `png`, `gif`, `bmp`。

**輸出內容**：

- 標記結果儲存於 `result.csv`，格式如下：
    - `Image`：圖片檔案名稱。
    - `Image_NoExt`：圖片檔案名稱（不含副檔名）。
    - `EMOTION_TAGS`、`ATTITUDE_TAGS`、`INTERACTION_TAGS`、`TONE_TAGS` 分別對應不同類別的標籤，標籤間以逗號 `,` 分隔。

**注意事項：**

- 若 `result.csv` 已存在，程式會讀取並載入已標記的結果。
- 若 `IMG` 內無圖片，程式會提示使用者新增圖片。
- 若圖片標記完成，程式會顯示完成訊息。

## LLM_Answer_Selector.py

**LLM答案選擇器**：根據情境，利用 GPT-4o-mini 模型從使用者提供的選項中選出最適合的答案，並將結果存入 CSV 檔案。

**主要功能包括**：

- 根據每個問題及其對應的選項，程式會請 GPT-4o-mini 做選擇題。選項會包括 A 到 F 六個選項，其中 F 代表「以上選項皆不符合」。

**輸入格式：**

- **`predictions.csv`**：程式會從此檔案中讀取問題及其對應選項。每一行必須包含以下欄位：
    - `title`：問題或對話的文字。
    - `label_1` 到 `label_5`：對應的五個選項，分別對應 A 到 E 的選項。
- **`openai_api_key.txt`**：此檔案應包含一個有效的 OpenAI API 密鑰。

**輸出格式：**

- **`ans.csv`**：儲存每個問題的回應結果，格式如下：
    - `title`：問題或對話的文字。
    - `answer`：選擇的回應（對應 A 到 F 中的某一個選項）。

**注意事項：**

- 若 `ans.csv` 已經存在，程式會將新的答案附加在檔案後面。
- 程式對 API 請求有延遲控制，以防止過度發送請求。

## Dialogue_Validation.py

**LLM答案驗證器**：利用GPT-4o 模型來判斷給定的對話內容是否"成立"，並將結果存入 CSV 檔案。此外，也會儲存 LLM 的原始回應，以便檢查錯誤。

**主要功能包括**：

- 程式從 `ans.csv` 讀取已標註的問題與答案，將每個問題與選項組合形成對話，並傳送給 OpenAI 模型進行分析。
- 模型回應的格式為每題的判斷結果，程式將這些回應解析為「成立」或「不成立」，並將結果儲存於 `results.csv`。

**輸入格式：**

- **`ans.csv`**：程式會從此檔案中讀取已標註的問題與答案，格式如下：
    - `title`：問題或對話的文字。
    - `answer`：對應的答案。
- **`openai_api_key.txt`**：此檔案必須包含有效的 OpenAI API 密鑰。

**輸出格式：**

- **`results.csv`**：儲存每個問題的回應結果，格式如下：
    - `title`：問題或對話的文字。
    - `answer`：對應的答案。
    - `LLMoutput`：模型回應的判斷結果（「成立」或「不成立」）。
- **`llmoutput.txt`**：儲存原始的 LLM 回應內容。

注意事項：

- 程式一次處理 10 條對話，並且每處理一批後會暫停 1 秒。
- 若 `results.csv` 已經存在，程式會將新的結果附加在檔案後面。

## Mygo_Model_Fine-tuning.py

**MyGO模型微調**：微調一個基於 RoBERTa 的中文對話模型。模型將用於中文文本分類任務，並根據標籤(MyGO台詞)對其進行預測。

**訓練方式：**

- **資料處理：**
    - 資料集來自兩個 JSON 檔案：`Label_Path.json` (標籤) 和 `dataset.json` (包含輸入文本與對應標籤)。
    - 將訓練測試資料切成8:2
- **模型訓練：**
    - 使用了 **混合精度訓練** (`autocast` 和 `GradScaler`) 來加速計算並降低 GPU 記憶體使用量。
    - 使用 **梯度累積** (`accumulation_steps = 2`) 來更新模型參數，這樣每隔兩個 mini-batches 才會進行一次權重更新，適合小批次訓練。
    - 每個 epoch 結束後，會計算並記錄訓練集和驗證集的平均損失和準確度。
    - 如果 5 個 epoch `avg_val_loss` 沒有比之前高，將會Early Stopping。

**模型組成：**

- 模型基於 **RoBERTa** (中文預訓練模型 `hfl/chinese-roberta-wwm-ext-large`) 並進行微調 (fine-tuning) 用於文本分類任務。
- **模型結構：**
    - **RoBERTa 架構：** RoBERTa 是一種基於 BERT 的語言模型，具有較大的語料庫和更多的訓練次數。這個模型被設計為適應長文本，並能夠捕捉文本的深層語義結構。
    - **分類層** : 該模型將 RoBERTa 的輸出送入一個分類層，並使用 softmax 函數將其轉化為各個標籤的概率分佈。

**模型參數：**

- **Tokenizer：** 使用 `AutoTokenizer` 來處理文本，將文本轉換為模型可以理解的格式，這裡使用的是中文的 RoBERTa 模型 (`hfl/chinese-roberta-wwm-ext-large`)。
- **學習率：** 使用 `AdamW` 優化器來進行模型優化，學習率設置為 `1e-5`。
- **損失函數：** 使用 **Label Smoothing Loss**，這是一種對真實標籤進行平滑處理的損失函數，能夠防止模型過擬合，尤其是當標籤類別不平衡時。
- **梯度裁剪：** 使用 `torch.nn.utils.clip_grad_norm_` 來限制梯度的範圍，防止梯度爆炸。
- **學習率調度器：Cosine Annealing Warm Restarts** 調度器將學習率根據訓練進度調整，並根據訓練過程中的熱重啟機制進行調整。

**輸入格式：**

- **`dataset.json`**：訓練數據集，包括每個對話的輸入文本與對應的標籤。格式範例：
    
    ```python
    [
      {"Input": "每天都像打仗一樣。", "Output": "人生這麼漫長會撐不住的喔"},
      {"Input": "我已經下定決心了！", "Output": "一旦加入就無法回頭了喔"}
    ]
    ```
    
- **`Label_Path.json`**：包含標籤的映射(用於訓練)以及對應網址(訓練中不會用到)，指明每個標籤的名稱。格式範例：
    
    ```python
    [
      {"title": "人生這麼漫長會撐不住的喔", "Image_Path": "http://xxx.jpg"},
      {"title": "一旦加入就無法回頭了喔。", "Image_Path": "http://xxx.jpg"}
    ]
    ```
    

**輸出格式：**

- **模型**：訓練完成後，模型參數將保存在 `models` 目錄中。

## test_model.py

**模型測試器**：測試訓練好的模型。

**主要功能包括**：

- **預測 (predict)**：基於輸入文本返回最可能的預測標籤及信心值。
- **多標籤預測 (predict_with_top_k)**：基於輸入文本返回前 N 個最可能的預測標籤及其信心值。

**命令行執行**：

```python
#格式：
python test_model.py "要預測的文本" num
#num為要預測的N個標籤數量
```

```python
#範例1
python test_model.py "明天孤獨搖滾開播" 1
#輸出
"""
最佳預測標籤: 要不要過去看看
信心值: 19.10%
"""
```

```python
#範例2
python test_model.py "明天孤獨搖滾開播" 3
#輸出
"""
前 3 個可能的預測：
1. 要不要過去看看 (信心值: 19.10%)
2. 那真是可喜可賀 (信心值: 10.09%)
3. 我也很想去喔 (信心值: 5.12%) 
"""
```

注意事項：

- 預測模型的輸入文本長度不能超過 `max_length`，預設為 512，若文本長度超過此限制，將會自動截斷。
- 信心值是基於模型預測結果的概率，範圍介於 `0.0` 到 `1.0`。
- 需要同目錄下放置models資料夾