import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

with open('negative_trainingSet.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = [line.strip() for line in f if line.strip()]
negative_df = pd.DataFrame({'title': lines})

with open('positive_trainingSet.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = [line.strip() for line in f if line.strip()]
positive_df = pd.DataFrame({'title': lines})
positive_df['label'] = 1
negative_df['label'] = 0

train_df = pd.concat([positive_df, negative_df], ignore_index=True)


test_df = pd.read_csv('testSet-1000.csv')


vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_df['title'])
y_train = train_df['label']

X_test = vectorizer.transform(test_df['title given by manchine'].fillna(''))
y_test = test_df['Y/N'].map({'Y':1, 'N':0})


nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)


y_pred = nb_classifier.predict(X_test)


print("Naive Bayes Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

svd = TruncatedSVD(n_components=50, random_state=42)
X_train_50 = svd.fit_transform(X_train)
X_test_50 = svd.transform(X_test)

# 2) 训练集与测试集合并后做 t-SNE 到二维
X_all_50 = np.vstack([X_train_50, X_test_50])
tsne = TSNE(
    n_components=2,
    perplexity=30,
    init='pca',
    learning_rate='auto',
    n_iter=1000,
    random_state=42,
    verbose=1
)
X_all_2d = tsne.fit_transform(X_all_50)

n_train = X_train_50.shape[0]
train_2d = X_all_2d[:n_train]
test_2d = X_all_2d[n_train:]

y_train_np = np.asarray(y_train)
y_test_np = np.asarray(y_test)
correct = (np.asarray(y_pred) == y_test_np)

# 3) 画图
plt.figure(figsize=(12, 5), dpi=140)

ax1 = plt.subplot(1, 2, 1)
for lbl, color, name in [(0, '#d62728', 'Neg'), (1, '#2ca02c', 'Pos')]:
    m_train = (y_train_np == lbl)
    m_test = (y_test_np == lbl)
    ax1.scatter(train_2d[m_train, 0], train_2d[m_train, 1],
                s=12, c=color, alpha=0.6, label=f'Train-{name}', marker='o')
    ax1.scatter(test_2d[m_test, 0], test_2d[m_test, 1],
                s=20, c=color, alpha=0.9, label=f'Test-{name}', marker='x')
ax1.set_title('t-SNE by true label')
ax1.legend(frameon=False, fontsize=9)
ax1.set_xticks([]); ax1.set_yticks([])

# 右：测试集预测对错（绿=正确，橙=错误）
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(test_2d[correct, 0], test_2d[correct, 1],
            s=20, c='#2ca02c', alpha=0.9, label='Correct', marker='o')
ax2.scatter(test_2d[~correct, 0], test_2d[~correct, 1],
            s=30, c='#ff7f0e', alpha=0.9, label='Wrong', marker='x')
ax2.set_title('t-SNE on test by prediction correctness')
ax2.legend(frameon=False, fontsize=9)
ax2.set_xticks([]); ax2.set_yticks([])

plt.tight_layout()
plt.show()