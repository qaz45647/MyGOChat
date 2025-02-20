import json
import torch
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DialoguePredictor:
    def __init__(self, model_path='models'):
        # 載入標籤映射
        with open(f'{model_path}/label_mapping.json', 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            self.idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}

        # 設置設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 載入分詞器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 載入訓練好的模型
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, return_confidence=False, max_length=512):
        # 對輸入文本進行編碼
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 將輸入移至正確的設備
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 使用混合精度推理
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

                # 獲取最可能的預測
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()

        # 獲取預測標籤
        predicted_label = self.idx_to_label[predicted_idx]

        if return_confidence:
            return predicted_label, confidence
        return predicted_label

    def predict_with_top_k(self, text, k=3, max_length=512):
        # 編碼輸入文本
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 將輸入移至正確的設備
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.cuda.amp.autocast():
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


if __name__ == '__main__':
    # 獲取命令行參數
    if len(sys.argv) < 3:
        print("請提供文本和獲取預測的數字。用法：python test_model.py '輸入文本' N")
        sys.exit(1)

    # 解析命令行參數
    text = sys.argv[1]
    k = int(sys.argv[2])


    # 初始化預測器
    predictor = DialoguePredictor('models')

    if k == 1:
        # 獲取單一最佳預測
        label, confidence = predictor.predict(text, return_confidence=True)
        print(f"最佳預測標籤: {label}")
        print(f"信心值: {confidence:.2%}")
    elif k > 1:
        # 獲取前 N 個最可能的預測
        top_k_predictions = predictor.predict_with_top_k(text, k)
        print(f"前 {k} 個可能的預測：")
        for i, (label, conf) in enumerate(top_k_predictions, 1):
            print(f"{i}. {label} (信心值: {conf:.2%})")
    else:
        print("N 必須是正數。")
