#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word2Vec + LibSVM 分类器
- 合并正负样本生成训练 CSV
- 训练 Word2Vec
- 使用 LibSVM 官方库训练 SVM
- 输出 Accuracy 和分类报告
"""

import pandas as pd
import numpy as np
import argparse
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from libsvm.svmutil import svm_train, svm_predict, svm_problem, svm_parameter
from data_preprocess import simple_tokenize

# ----------------------------
# 合并正负样本生成 CSV
# ----------------------------
# 读取正负样本
with open('positive_trainingSet.txt', 'r', encoding='utf-8') as f:
    pos_lines = [line.strip() for line in f if line.strip()]

with open('negative_trainingSet.txt', 'r', encoding='utf-8') as f:
    neg_lines = [line.strip() for line in f if line.strip()]

# 生成 DataFrame
df_pos = pd.DataFrame({'title': pos_lines, 'label': 1})
df_neg = pd.DataFrame({'title': neg_lines, 'label': 0})

# 合并
df_train = pd.concat([df_pos, df_neg], ignore_index=True)

# 保存 CSV
df_train.to_csv('train.csv', index=False, encoding='utf-8')
print("训练 CSV 已生成: train.csv")

# ----------------------------
# 文本 -> 平均词向量
# ----------------------------
def get_avg_vector(w2v_model, tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

# ----------------------------
# 主函数
# ----------------------------
def main(args):
    # 1️⃣ 合并正负样本生成 CSV
    train_csv = merge_samples(args.pos_file, args.neg_file, args.output_csv)

    # 2️⃣ 读取训练 CSV
    df = pd.read_csv(train_csv)
    df['tokens'] = df[args.text_col].apply(simple_tokenize)
    texts = df['tokens']
    labels = df[args.label_col].astype(int).values

    # 3️⃣ 划分训练/测试集
    if args.test_csv:
        df_test = pd.read_csv(args.test_csv)
        df_test['tokens'] = df_test[args.text_col].apply(simple_tokenize)
        X_train_tokens, X_test_tokens = texts, df_test['tokens']
        y_train, y_test = labels, df_test[args.label_col].astype(int).values
    else:
        X_train_tokens, X_test_tokens, y_train, y_test = train_test_split(
            texts, labels, test_size=args.test_size, random_state=42, stratify=labels
        )

    print("训练集样本数:", len(X_train_tokens))
    print("测试集样本数:", len(X_test_tokens))

    # 4️⃣ 训练 Word2Vec
    if args.load_w2v and args.load_w2v.strip():
        print(f"加载 Word2Vec 模型: {args.load_w2v}")
        w2v_model = Word2Vec.load(args.load_w2v)
    else:
        print("训练 Word2Vec 模型...")
        w2v_model = Word2Vec(
            sentences=X_train_tokens.tolist(),
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            sg=1 if args.skipgram else 0,
            workers=args.workers,
            epochs=args.epochs
        )
        if args.save_w2v:
            w2v_model.save(args.save_w2v)
            print(f"Word2Vec 模型已保存: {args.save_w2v}")

    # 5️⃣ 文本 -> 平均词向量
    X_train_vec = np.array([get_avg_vector(w2v_model, tokens) for tokens in X_train_tokens])
    X_test_vec = np.array([get_avg_vector(w2v_model, tokens) for tokens in X_test_tokens])

    # 6️⃣ 标准化
    scaler = StandardScaler()
    X_train_vec = scaler.fit_transform(X_train_vec)
    X_test_vec = scaler.transform(X_test_vec)

    # 7️⃣ LibSVM 训练
    print("训练 LibSVM 模型...")
    train_data = svm_problem(y_train.tolist(), X_train_vec.tolist())
    param = svm_parameter(args.libsvm_params)
    model = svm_train(train_data, param)

    # 8️⃣ 预测与评估
    print("预测测试集...")
    p_label, p_acc, p_vals = svm_predict(y_test.tolist(), X_test_vec.tolist(), model)

    print("\n===== Word2Vec + LibSVM 结果 =====")
    print("Accuracy:", accuracy_score(y_test, p_label))
    print(classification_report(y_test, p_label, digits=4))

# ----------------------------
# argparse 参数
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Word2Vec + LibSVM 分类器")
    parser.add_argument("--pos-file", type=str, default="data/positive_trainingSet.txt", help="正样本文件")
    parser.add_argument("--neg-file", type=str, default="data/negative_trainingSet.txt", help="负样本文件")
    parser.add_argument("--output-csv", type=str, default="data/train.csv", help="生成训练 CSV 文件")
    parser.add_argument("--test-csv", type=str, default="", help="可选测试 CSV")
    parser.add_argument("--text-col", type=str, default="title", help="文本列名")
    parser.add_argument("--label-col", type=str, default="label", help="标签列名")
    parser.add_argument("--test-size", type=float, default=0.2, help="无测试集时划分比例")
    parser.add_argument("--vector-size", type=int, default=100, help="Word2Vec向量维度")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec窗口大小")
    parser.add_argument("--min-count", type=int, default=2, help="Word2Vec最小词频")
    parser.add_argument("--skipgram", action="store_true", help="使用Skip-gram，否则CBOW")
    parser.add_argument("--epochs", type=int, default=20, help="Word2Vec训练轮数")
    parser.add_argument("--workers", type=int, default=4, help="Word2Vec并行线程数")
    parser.add_argument("--save-w2v", type=str, default="models/w2v.model", help="保存Word2Vec模型路径")
    parser.add_argument("--load-w2v", type=str, default="", help="加载已有Word2Vec模型路径")
    parser.add_argument("--libsvm-params", type=str, default='-t 2 -c 1 -g 0.01', help="LibSVM参数")
    args = parser.parse_args()
    main(args)
