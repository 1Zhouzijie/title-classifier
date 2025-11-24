import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import umap

# ----------------------
# 1️⃣ 加载训练数据
# ----------------------
print("[Step 1] Loading training data...")
with open('Word2Vec_SVM/data/negative_trainingSet.txt', 'r', encoding='utf-8', errors='ignore') as f:
    neg_lines = [line.strip() for line in f if line.strip()]
negative_df = pd.DataFrame({'title': neg_lines, 'label': 0})

with open('Word2Vec_SVM/data/positive_trainingSet.txt', 'r', encoding='utf-8', errors='ignore') as f:
    pos_lines = [line.strip() for line in f if line.strip()]
positive_df = pd.DataFrame({'title': pos_lines, 'label': 1})

train_df = pd.concat([positive_df, negative_df], ignore_index=True)
print("[Step 1] Training data loaded ✅")

# ----------------------
# 2️⃣ 加载测试数据
# ----------------------
print("[Step 2] Loading test data...")
test_df = pd.read_excel('Word2Vec_SVM/data/testSet-1000.xlsx')
test_df['label'] = test_df['Y/N'].map({'Y':1, 'N':0})
print("[Step 2] Test data loaded ✅")

# ----------------------
# 3️⃣ 分词（去停用词）
# ----------------------
def tokenize_no_stopwords(text):
    if not isinstance(text, str):
        return []
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]

print("[Step 3] Tokenizing titles...")
train_df['tokens'] = train_df['title'].apply(tokenize_no_stopwords)
test_df['tokens'] = test_df['title given by manchine'].fillna('').apply(tokenize_no_stopwords)
print("[Step 3] Tokenizing completed ✅")

# ----------------------
# 4️⃣ Word2Vec 训练
# ----------------------
print("[Step 4] Training Word2Vec model...")
w2v_model = Word2Vec(
    sentences=train_df['tokens'].tolist(),
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=30
)
print("[Step 4] Word2Vec training completed ✅")

# ----------------------
# 5️⃣ 句子向量化（平均词向量）
# ----------------------
def sentence_vector(tokens):
    vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

print("[Step 5] Converting sentences to vectors...")
X_train = np.vstack(train_df['tokens'].apply(sentence_vector))
y_train = train_df['label'].values
X_test = np.vstack(test_df['tokens'].apply(sentence_vector))
y_test = test_df['label'].values
print("[Step 5] Sentence vectors ready ✅")

# ----------------------
# 6️⃣ 标准化
# ----------------------
print("[Step 6] Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("[Step 6] Standardization completed ✅")

# ----------------------
# 7️⃣ SVM 分类
# ----------------------
print("[Step 7] Training SVM classifier...")
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
print("[Step 7] SVM training completed ✅")

# ----------------------
# 8️⃣ 输出指标
# ----------------------
print("SVM Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# ----------------------
# 9️⃣ SVD + UMAP 可视化
# ----------------------
print("[Step 9] Dimensionality reduction for visualization...")
svd = TruncatedSVD(n_components=50, random_state=42)
X_train_50 = svd.fit_transform(X_train)
X_test_50 = svd.transform(X_test)

umap_model = umap.UMAP(
    n_components=2,      # 2D 可视化
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
    n_jobs=-1
)
train_2d = umap_model.fit_transform(X_train_50)
test_2d = umap_model.transform(X_test_50)

correct = (y_pred == y_test)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(train_2d[y_train==0,0], train_2d[y_train==0,1], c='lightcoral', alpha=0.3, label='Train-Neg')
plt.scatter(train_2d[y_train==1,0], train_2d[y_train==1,1], c='#2ca02c', alpha=0.7, label='Train-Pos')
plt.scatter(test_2d[y_test==0,0], test_2d[y_test==0,1], c='lightcoral', alpha=0.5, label='Test-Neg', marker='x')
plt.scatter(test_2d[y_test==1,0], test_2d[y_test==1,1], c='#2ca02c', alpha=0.9, label='Test-Pos', marker='x')
plt.title('UMAP(2D) by true label')
plt.legend(frameon=False, fontsize=9)

plt.subplot(1,2,2)
plt.scatter(test_2d[correct,0], test_2d[correct,1], c='#2ca02c', label='Correct', alpha=0.9)
plt.scatter(test_2d[~correct,0], test_2d[~correct,1], c='#ff7f0e', label='Wrong', alpha=0.9)
plt.title('UMAP(2D) on test by prediction correctness')
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()
print("[Step 9] Visualization completed ✅")








