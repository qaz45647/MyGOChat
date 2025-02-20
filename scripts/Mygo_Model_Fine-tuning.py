import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import sys
from torch.cuda.amp import autocast, GradScaler

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class DialogueDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler

def train_model(
    train_loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    tokenizer,
    device,
    num_epochs,
    patience=5,
    save_path='models'
):
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0
    accumulation_steps = 2
    
    model.gradient_checkpointing_enable()
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for i, batch in enumerate(train_progress):
            with autocast():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * accumulation_steps
            train_progress.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        
        # 驗證階段
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_val_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                val_progress.set_postfix({
                    'loss': f'{outputs.loss.item():.4f}',
                    'acc': f'{100*correct/total:.2f}%'
                })
        
        # 計算平均損失
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # 輸出訓練統計
        logging.info(f'\nEpoch {epoch+1}/{num_epochs}:')
        logging.info(f'Average training loss: {avg_train_loss:.4f}')
        logging.info(f'Average validation loss: {avg_val_loss:.4f}')
        logging.info(f'Validation accuracy: {accuracy:.2f}%')
        
        # Early stopping 檢查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logging.info(f'Saved best model to {save_path}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        logging.info('-' * 50)

def main():
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    try:
        # 載入數據
        with open('Label_Path.json', 'r', encoding='utf-8') as f:
            labels = json.load(f)
        with open('dataset.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 準備標籤映射
        label_to_idx = {label['title']: idx for idx, label in enumerate(labels)}
        idx_to_label = {idx: label['title'] for idx, label in enumerate(labels)}
        
        # 分割數據
        train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
        
        # 準備數據集
        tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
        
        train_labels = [label_to_idx[item['Output']] for item in train_data]
        train_dataset = DialogueDataset(
            [item['Input'] for item in train_data],
            train_labels,
            tokenizer
        )
        val_dataset = DialogueDataset(
            [item['Input'] for item in val_data],
            [label_to_idx[item['Output']] for item in val_data],
            tokenizer
        )
        
        # 創建加權採樣器
        train_sampler = create_weighted_sampler(train_labels)
        
        # 創建數據加載器
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            sampler=train_sampler,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            num_workers=2
        )
        
        # 初始化模型
        model = AutoModelForSequenceClassification.from_pretrained(
            'hfl/chinese-roberta-wwm-ext-large',
            num_labels=len(label_to_idx),
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            problem_type="single_label_classification"
        )
        model.to(device)
        
        # 設置分層學習率
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
        
        # 設置學習率調度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=len(train_loader) * 2,
            T_mult=2
        )
        
        # 設置損失函數
        criterion = LabelSmoothingLoss(classes=len(label_to_idx), smoothing=0.1)
        
        # 開始訓練
        train_model(
            train_loader,
            val_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            tokenizer,
            device,
            num_epochs=100,
            patience=5
        )
        
        # 保存標籤映射
        with open('models/label_mapping.json', 'w', encoding='utf-8') as f:
            json.dump({
                'label_to_idx': label_to_idx,
                'idx_to_label': idx_to_label
            }, f, ensure_ascii=False, indent=2)
        
        logging.info('Training completed successfully!')
        
    except Exception as e:
        logging.error(f'An error occurred: {str(e)}')
        raise

if __name__ == '__main__':
    main()