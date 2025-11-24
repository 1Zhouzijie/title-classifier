import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
import warnings
import os
import pickle
import json
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TitleDataset:
    def __init__(self, data_dir="D:/Codes/titles"):
        self.data_dir = data_dir
        self.positive_titles = []
        self.negative_titles = []
        
    def load_data(self):
        """加载正负样本数据"""
        # 加载正样本
        try:
            with open(f"{self.data_dir}/positive_trainingSet.txt", 'r', encoding='utf-8', errors='ignore') as f:
                self.positive_titles = [line.strip() for line in f if line.strip()]
        except:
            print("Error loading positive training set")
            
        # 加载负样本
        try:
            with open(f"{self.data_dir}/negative_trainingSet.txt", 'r', encoding='utf-8', errors='ignore') as f:
                self.negative_titles = [line.strip() for line in f if line.strip()]
        except:
            print("Error loading negative training set")
            
        print(f"Loaded {len(self.positive_titles)} positive samples")
        print(f"Loaded {len(self.negative_titles)} negative samples")
        
        return self.create_dataset()
    
    def create_dataset(self):
        """创建完整的数据集"""
        # 合并数据
        titles = self.positive_titles + self.negative_titles
        labels = [1] * len(self.positive_titles) + [0] * len(self.negative_titles)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'title': titles,
            'label': labels
        })
        
        # 随机打乱
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df

class BertDataset(Dataset):
    def __init__(self, titles, labels, tokenizer, max_length=128):
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, idx):
        title = str(self.titles[idx])
        label = self.labels[idx]
        
        # 编码文本
        encoding = self.tokenizer(
            title,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertSVMClassifier:
    """方法1: BERT编码 + SVM分类"""
    
    def __init__(self, model_name='bert-base-uncased', model_dir='./models'):
        self.model_name = model_name
        self.model_dir = model_dir
        self.tokenizer = None
        self.bert_model = None
        self.svm_classifier = None
        
        # 创建模型保存目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 定义模型文件路径
        self.svm_model_path = os.path.join(model_dir, 'bert_svm_classifier.pkl')
        self.bert_model_path = os.path.join(model_dir, 'bert_model')
        self.config_path = os.path.join(model_dir, 'bert_svm_config.json')
    
    def _load_bert_model(self):
        """加载BERT模型"""
        if self.tokenizer is None or self.bert_model is None:
            print(f"Loading BERT model: {self.model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.bert_model = BertModel.from_pretrained(self.model_name)
            self.bert_model.to(device)
            self.bert_model.eval()
    
    def clean_titles(self, titles):
        """清理标题数据"""
        cleaned = []
        for title in titles:
            if title is None or pd.isna(title):
                cleaned.append("empty title")
            else:
                title_str = str(title).strip()
                if not title_str:
                    cleaned.append("empty title")
                else:
                    cleaned.append(title_str)
        return cleaned
        
    def get_bert_embeddings(self, titles, batch_size=16):
        """获取BERT词向量"""
        self._load_bert_model()
        
        # 清理数据
        clean_titles = self.clean_titles(titles)
        print(f"Processing {len(clean_titles)} titles for embeddings...")

        embeddings = []
        
        for i in range(0, len(clean_titles), batch_size):
            batch_titles = clean_titles[i:i+batch_size]
            
            # 编码文本
            encoded = self.tokenizer(
                batch_titles,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # 获取BERT输出
            with torch.no_grad():
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                # 使用[CLS]标记的输出作为句子表示
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.extend(cls_embeddings.cpu().numpy())
                
        return np.array(embeddings)
    
    def save_model(self):
        """保存模型"""
        print(f"Saving BERT+SVM model to {self.model_dir}")
        
        # 保存SVM分类器
        if self.svm_classifier is not None:
            with open(self.svm_model_path, 'wb') as f:
                pickle.dump(self.svm_classifier, f)
        
        # 保存BERT模型（可选，因为可以直接从HuggingFace加载）
        if self.bert_model is not None and self.tokenizer is not None:
            self.bert_model.save_pretrained(self.bert_model_path)
            self.tokenizer.save_pretrained(self.bert_model_path)
        
        # 保存配置信息
        config = {
            'model_name': self.model_name,
            'model_type': 'BertSVMClassifier'
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("Model saved successfully!")
    
    def load_model(self):
        """加载已保存的模型"""
        if not os.path.exists(self.svm_model_path):
            print("No saved SVM model found. Need to train first.")
            return False
        
        try:
            print(f"Loading BERT+SVM model from {self.model_dir}")
            
            # 加载SVM分类器
            with open(self.svm_model_path, 'rb') as f:
                self.svm_classifier = pickle.load(f)
            
            # 加载BERT模型
            if os.path.exists(self.bert_model_path):
                self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
                self.bert_model = BertModel.from_pretrained(self.bert_model_path)
            else:
                # 如果没有保存，从HuggingFace加载
                self._load_bert_model()
            
            self.bert_model.to(device)
            self.bert_model.eval()
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def train(self, train_titles, train_labels, save_after_training=True):
        """训练模型"""
        print("Getting BERT embeddings for training data...")
        self._load_bert_model()
        train_embeddings = self.get_bert_embeddings(train_titles)
        
        print("Training SVM classifier...")
        self.svm_classifier = SVC(kernel='rbf', C=1.0, random_state=42)
        self.svm_classifier.fit(train_embeddings, train_labels)
        
        if save_after_training:
            self.save_model()
        
    def predict(self, test_titles):
        """预测"""
        if self.svm_classifier is None:
            raise ValueError("Model not trained or loaded. Please train or load model first.")
        
        print("Getting BERT embeddings for test data...")
        test_embeddings = self.get_bert_embeddings(test_titles)
        
        return self.svm_classifier.predict(test_embeddings)

class BertForClassification:
    """方法2: BertForSequenceClassification"""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2, model_dir='./models'):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        
        # 创建模型保存目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 定义模型文件路径
        self.bert_classifier_path = os.path.join(model_dir, 'bert_classifier')
        self.config_path = os.path.join(model_dir, 'bert_classifier_config.json')
    
    def _load_model(self):
        """初始化模型"""
        if self.tokenizer is None or self.model is None:
            print(f"Loading BERT classification model: {self.model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=self.num_labels
            )
            self.model.to(device)
    
    def save_model(self):
        """保存模型"""
        print(f"Saving BertForSequenceClassification model to {self.model_dir}")
        
        if self.model is not None and self.tokenizer is not None:
            self.model.save_pretrained(self.bert_classifier_path)
            self.tokenizer.save_pretrained(self.bert_classifier_path)
        
        # 保存配置信息
        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'model_type': 'BertForSequenceClassification'
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("Model saved successfully!")
    
    def load_model(self):
        """加载已保存的模型"""
        if not os.path.exists(self.bert_classifier_path):
            print("No saved BertForSequenceClassification model found. Need to train first.")
            return False
        
        try:
            print(f"Loading BertForSequenceClassification model from {self.model_dir}")
            
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_classifier_path)
            self.model = BertForSequenceClassification.from_pretrained(self.bert_classifier_path)
            self.model.to(device)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
    def train(self, train_titles, train_labels, val_titles, val_labels, 
              epochs=3, batch_size=16, learning_rate=2e-5, save_after_training=True):
        """训练模型"""
        self._load_model()
        
        # 创建数据集
        train_dataset = BertDataset(train_titles, train_labels, self.tokenizer)
        val_dataset = BertDataset(val_titles, val_labels, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 优化器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # 训练
        self.model.train()
        best_val_accuracy = 0.0
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(input_ids=input_ids, 
                                   attention_mask=attention_mask, 
                                   labels=labels)
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 验证
            val_accuracy = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {total_loss/len(train_loader):.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}")
            
            # 保存最佳模型
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if save_after_training:
                    self.save_model()
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                predictions.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())
        
        return accuracy_score(actual_labels, predictions)
    
    def predict(self, test_titles):
        """预测"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load model first.")
            
        test_labels = [0] * len(test_titles)  # 占位符
        test_dataset = BertDataset(test_titles, test_labels, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        return np.array(predictions)

def load_test_data(test_file="D:/Codes/titles/testSet-1000.csv"):
    """加载测试数据"""
    try:
        df = pd.read_csv(test_file)
        # 根据Y/N列创建标签，Y=1, N=0
        df['label'] = (df['Y/N'] == 'Y').astype(int)
        return df['title given by manchine'].tolist(), df['label'].tolist()
    except Exception as e:
        print(f"Error loading test data: {e}")
        return [], []

def evaluate_predictions(y_true, y_pred, method_name):
    """评估预测结果"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print(f"\n=== {method_name} Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Poor Title', 'Good Title']))
    
    return accuracy, precision, recall, f1

def main():
    print("=== BERT-based Title Classification ===\n")
    
    # 设置是否重新训练模型
    retrain_models = input("Do you want to retrain models? (y/n, default: n): ").lower().strip()
    retrain_models = retrain_models == 'y'
    
    # 1. 加载数据
    print("Loading dataset...")
    dataset = TitleDataset()
    df = dataset.load_data()
    
    if len(df) == 0:
        print("No data loaded. Please check your data files.")
        return
    
    # 2. 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_titles = train_df['title'].tolist()
    train_labels = train_df['label'].tolist()
    val_titles = val_df['title'].tolist()
    val_labels = val_df['label'].tolist()
    
    print(f"Training set: {len(train_titles)} samples")
    print(f"Validation set: {len(val_titles)} samples")
    
    # 3. 加载测试数据
    test_titles, test_labels = load_test_data()
    if len(test_titles) > 0:
        print(f"Test set: {len(test_titles)} samples")
    else:
        print("No test data found, using validation set for final evaluation")
        test_titles, test_labels = val_titles, val_labels
    
    # 4. 方法1: BERT + SVM
    print("\n" + "="*50)
    print("Method 1: BERT Embeddings + SVM")
    print("="*50)
    
    bert_svm = BertSVMClassifier()
    
    if not retrain_models and bert_svm.load_model():
        print("Loaded existing BERT+SVM model")
    else:
        print("Training BERT+SVM model...")
        bert_svm.train(train_titles, train_labels)
    
    svm_predictions = bert_svm.predict(test_titles)
    svm_results = evaluate_predictions(test_labels, svm_predictions, "BERT + SVM")
    
    # 5. 方法2: BertForSequenceClassification
    print("\n" + "="*50)
    print("Method 2: BertForSequenceClassification")
    print("="*50)
    
    bert_classifier = BertForClassification()
    
    if not retrain_models and bert_classifier.load_model():
        print("Loaded existing BertForSequenceClassification model")
    else:
        print("Training BertForSequenceClassification model...")
        bert_classifier.train(train_titles, train_labels, val_titles, val_labels)
    
    bert_predictions = bert_classifier.predict(test_titles)
    bert_results = evaluate_predictions(test_labels, bert_predictions, "BertForSequenceClassification")
    
    # 6. 比较结果
    print("\n" + "="*50)
    print("COMPARISON OF METHODS")
    print("="*50)
    print(f"{'Metric':<15} {'BERT+SVM':<12} {'BERT Classifier':<15}")
    print("-" * 45)
    print(f"{'Accuracy':<15} {svm_results[0]:<12.4f} {bert_results[0]:<15.4f}")
    print(f"{'Precision':<15} {svm_results[1]:<12.4f} {bert_results[1]:<15.4f}")
    print(f"{'Recall':<15} {svm_results[2]:<12.4f} {bert_results[2]:<15.4f}")
    print(f"{'F1-Score':<15} {svm_results[3]:<12.4f} {bert_results[3]:<15.4f}")
    
    # 7. 保存一些预测示例
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    for i in range(min(10, len(test_titles))):
        title = test_titles[i][:80] + "..." if len(test_titles[i]) > 80 else test_titles[i]
        print(f"\nTitle: {title}")
        print(f"True Label: {'Good' if test_labels[i] == 1 else 'Poor'}")
        print(f"BERT+SVM: {'Good' if svm_predictions[i] == 1 else 'Poor'}")
        print(f"BERT Classifier: {'Good' if bert_predictions[i] == 1 else 'Poor'}")

if __name__ == "__main__":
    main()