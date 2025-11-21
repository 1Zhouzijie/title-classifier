import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import umap

# 读取训练数据
with open('negative_trainingSet.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = [line.strip() for line in f if line.strip()]
negative_df = pd.DataFrame({'title': lines})

with open('positive_trainingSet.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = [line.strip() for line in f if line.strip()]
positive_df = pd.DataFrame({'title': lines})

positive_df['label'] = 1
negative_df['label'] = 0
train_df = pd.concat([positive_df, negative_df], ignore_index=True)

# 读取测试数据
test_df = pd.read_csv('testSet-1000.csv')

# 向量化（词袋）
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_df['title'])
y_train = train_df['label']

X_test = vectorizer.transform(test_df['title given by manchine'].fillna(''))
y_test = test_df['Y/N'].map({'Y': 1, 'N': 0})

# 朴素贝叶斯分类
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)

print("Naive Bayes Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# SVD 降维（作为 UMAP 输入）
svd = TruncatedSVD(n_components=50, random_state=42)
X_train_50 = svd.fit_transform(X_train)
X_test_50 = svd.transform(X_test)

# 使用 UMAP：在训练集上 fit，再对测试集 transform；三维可视化
umap_model = umap.UMAP(
    n_components=3,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',   # 文本表示上通常更稳
    random_state=42,
    n_jobs=-1
)
train_3d = umap_model.fit_transform(X_train_50)
test_3d = umap_model.transform(X_test_50)

y_train_np = np.asarray(y_train)
y_test_np = np.asarray(y_test)
correct = (np.asarray(y_pred) == y_test_np)

# 作图（3D）：左图按真实标签显示训练+测试；右图按测试预测对错
fig = plt.figure(figsize=(12, 5), dpi=140)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# 1. 先绘制 'Neg' 类别，并使用更低的 alpha 值
neg_color = 'lightcoral'
m_train_neg = (y_train_np == 0)
m_test_neg = (y_test_np == 0)
# 绘制训练集 'Neg' 点 (更透明)
ax1.scatter(train_3d[m_train_neg, 0], train_3d[m_train_neg, 1], train_3d[m_train_neg, 2],
            s=12, c=neg_color, alpha=0.3, label='Train-Neg', marker='o')
# 绘制测试集 'Neg' 点 (更透明)
ax1.scatter(test_3d[m_test_neg, 0], test_3d[m_test_neg, 1], test_3d[m_test_neg, 2],
            s=20, c=neg_color, alpha=0.5, label='Test-Neg', marker='x')

# 2. 后绘制 'Pos' 类别，使其位于顶层，并稍微增大点的大小
pos_color = '#2ca02c' # 绿色
m_train_pos = (y_train_np == 1)
m_test_pos = (y_test_np == 1)
# 绘制训练集 'Pos' 点
ax1.scatter(train_3d[m_train_pos, 0], train_3d[m_train_pos, 1], train_3d[m_train_pos, 2],
            s=15, c=pos_color, alpha=0.7, label='Train-Pos', marker='o')
# 绘制测试集 'Pos' 点
ax1.scatter(test_3d[m_test_pos, 0], test_3d[m_test_pos, 1], test_3d[m_test_pos, 2],
            s=25, c=pos_color, alpha=0.9, label='Test-Pos', marker='x')

ax1.set_title('UMAP(3D) by true label')
ax1.legend(frameon=False, fontsize=9)
ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(test_3d[correct, 0], test_3d[correct, 1], test_3d[correct, 2],
            s=20, c='#2ca02c', alpha=0.9, label='Correct', marker='o')
ax2.scatter(test_3d[~correct, 0], test_3d[~correct, 1], test_3d[~correct, 2],
            s=30, c='#ff7f0e', alpha=0.9, label='Wrong', marker='x')
ax2.set_title('UMAP(3D) on test by prediction correctness')
ax2.legend(frameon=False, fontsize=9)
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])

plt.tight_layout()
plt.show()