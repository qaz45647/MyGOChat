# 🐧MyGOChat


**MyGOChat** 是一個基於 [RoBERTa](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) 微調而成的模型，能夠自動分析輸入的文字內容，並推薦適合的 MyGO 圖。簡化你找 MyGO 圖的過程，更輕鬆的敷衍人。

<p align="center">
  <img src="https://truth.bahamut.com.tw/s01/202502/d02adc47c0be9dc4a0be1b9d662baa28.JPG" alt="Image Description" width="70%">
</p>



<h4 align="center">
  <a href="https://home.gamer.com.tw/artwork.php?sn=6093843">模型介紹</a> | 
  <a href="https://colab.research.google.com/drive/1boPRNFEXHklGYngmG4nSsydPa9pfF9jP?usp=sharing">測試模型</a>
</h4>

<p align="center">
  <img src="https://truth.bahamut.com.tw/s01/202511/bf7c26730e6236bcd511691320344738.JPG" alt="Image Description" width="100%">
</p>

# **Installation**


### **Install via GitHub**

**Clone the repo:**

```bash
git clone https://github.com/qaz45647/MyGOChat.git
```

```bash
cd MyGOChat
```

**Create a conda environment:**

```bash
conda create -n MyGO_env python=3.9
conda activate MyGO_env
```

**Use pip to install required packages:**

```bash
pip install -r requirements.txt
```

# **Usage**


```python
from mygochat import MyGOChat

# 初始化聊天
chat = MyGOChat()

# 使用聊天功能
response = chat.chat("輸入你要餵給模型的內容")

# 輸出回應
print(f"Quote: {response['quote']}")
print(f"Image URL: {response['image_url']}")


# 取出5個候選答案
result = chat.chat_with_candidates("輸入你要餵給模型的內容")
print(result)
print('\n')
#取出第二個結果
print(f"Quote: {result['candidates'][1]['quote']}")
print(f"Image URL: {result['candidates'][1]['image_url']}")

```

# **Acknowledgements**


本專案是基於 `chinese-roberta-wwm-ext-large` 模型進行開發。  
原始模型由哈工大訊飛聯合實驗室（HFL ) 提供 (https://github.com/ymcui/Chinese-BERT-wwm)

# **License**


![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

# **Other**


### **注意事項**

1. **語言限制：** 本模型僅針對繁體中文進行微調，若您使用其他語言，模型的表現和正確性無法得到保證。請謹慎使用其他語言進行測試。
2. **正確率：** 本模型的正確率為 86%，因此在某些情況下可能會出現答非所問或理解偏差的情況。請在使用過程中留意模型的回應，並視需要進行調整。此外，模型尚未在真實對話情境中進行測試，因此其實際效果仍有待驗證。
3. **圖片理解：** 本模型無法直接理解或分析圖片。如果您希望模型對圖片進行理解，請自行準備適合的多模態語言模型 (MLLM) 對圖片進行描述，並將描述進行適當的調整後，提供給模型以進行後續處理。
4. **多輪對話：** 本模型未經過針對多輪上下文的專門訓練，因此在進行多輪對話時，模型的上下文理解和連貫性可能會受到影響。使用者需注意並適時提供更多的背景訊息以確保回應的準確性。
5. **第一次執行：** 初次執行時會花較長時間載入模型和資源，後續執行速度會加快。

### **模型使用聲明**

感謝您使用本模型。本模型已開源並可用於商業用途，請您在使用時遵守以下條款：

1. **開源許可：** 本模型為開源軟體，您可以自由使用、修改、分發該模型的源代碼，前提是遵守本使用聲明及相應的 Apache 2.0 授權協議。
2. **商業用途：** 您可以將本模型應用於商業項目，包括但不限於產品開發、商業服務等。
3. **禁止用途：** 本模型不可用於以下用途：
    - 任何違反當地法律的活動或用途；
    - 生成或促進非法、惡意、詐騙、或有害的內容；
    - 用於侵犯他人知識產權、隱私權、或其他法律權益的行為。
4. **責任免責：** 本模型由開發者提供，但不保證其完美運行或不會出現任何錯誤。使用者在使用過程中應自行承擔風險，開發者不對任何因使用本模型所引發的直接或間接損失負責。
5. **版權聲明：** 本模型及其源代碼的版權歸開發者所有。使用者在使用過程中應遵循相應的開源協議條款。
6. **資料集聲明：** 訓練過程中因為有包含（爬取網路文章標題）等未經授權內容，故無法提供資料集，請見諒。
