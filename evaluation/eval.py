import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import csv
from pathlib import Path

class DialoguePredictor:
    def __init__(self, model_path='dialogue_model2'):
        print(f"正在載入模型從: {model_path}")
        # 載入標籤映射
        with open(f'{model_path}/label_mapping.json', 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            self.idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}
        
        # 設置設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {self.device}")
        
        # 載入分詞器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("模型載入完成")

    def predict_with_top_k(self, text, k=3, max_length=512):
        # 對輸入文本進行編碼，並明確設置 attention_mask
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        # 將所有輸入移至正確的設備
        inputs = {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
        
        # 使用 torch.amp.autocast() 進行混合精度推理
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                
                top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=1)
                
                results = []
                for i in range(k):
                    label = self.idx_to_label[top_k_indices[0][i].item()]
                    confidence = top_k_probs[0][i].item()
                    results.append((label, confidence))
        
        return results

def process_baha_data(predictor, input_file='baha.json', output_file='result.csv'):
    print(f"開始處理數據")
    print(f"輸入文件: {input_file}")
    print(f"輸出文件: {output_file}")
    
    # 確保輸出目錄存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 檢查輸入文件是否存在
    if not Path(input_file).exists():
        raise FileNotFoundError(f"找不到輸入文件: {input_file}")
    
    # 讀取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"成功讀取 {len(data)} 筆數據")
    
    # 創建新的 CSV 文件並寫入標題行
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Input', 'Prediction1', 'Prediction2', 'Prediction3'])
    
    processed_count = 0
    error_count = 0
    
    # 使用tqdm創建進度條
    for item in tqdm(data, desc="Processing titles"):
        try:
            title = item['title']
            predictions = predictor.predict_with_top_k(title, k=3)
            
            # 準備要寫入CSV的行
            row = [title]
            for pred, conf in predictions:
                row.append(f"{pred} ({conf:.2%})")
            
            # 將結果寫入CSV
            with open(output_file, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            processed_count += 1
            
            # 每處理100筆資料輸出一次進度
            if processed_count % 100 == 0:
                print(f"已處理 {processed_count} 筆資料")
            
        except Exception as e:
            error_count += 1
            print(f"\n處理標題時發生錯誤: {title}")
            print(f"錯誤信息: {str(e)}")
            continue
    
    print(f"\n處理完成！")
    print(f"成功處理: {processed_count} 筆")
    print(f"處理失敗: {error_count} 筆")
    print(f"結果已保存到: {output_file}")
    
    # 驗證輸出文件
    if Path(output_file).exists():
        with open(output_file, 'r', encoding='utf-8-sig') as f:
            line_count = sum(1 for line in f)
        print(f"輸出文件共有 {line_count} 行")
    else:
        print("警告：輸出文件未成功創建！")

if __name__ == '__main__':
    # 初始化預測器
    predictor = DialoguePredictor('dialogue_model2')
    
    # 處理數據
    process_baha_data(predictor)