import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from typing import List, Tuple

# ensure local package import works when run from anywhere
sys.path.append(os.path.dirname(__file__))

# import interfaces from your bert_new implementation
from bert_new import TitleDataset, BertSVMClassifier, BertForClassification

def _subsample_balanced(titles: List[str], labels: List[int], max_per_class: int = 500, random_state: int = 42
                       ) -> Tuple[List[str], List[int]]:
    np.random.seed(random_state)
    titles = np.array(titles)
    labels = np.array(labels)
    kept_idx = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) > max_per_class:
            idx = np.random.choice(idx, max_per_class, replace=False)
        kept_idx.extend(idx.tolist())
    kept_idx = np.array(sorted(kept_idx))
    return titles[kept_idx].tolist(), labels[kept_idx].tolist()

def visualize_classification(bert_svm: BertSVMClassifier,
                             bert_clf: BertForClassification,
                             train_titles: List[str],
                             train_labels: List[int],
                             test_titles: List[str],
                             test_labels: List[int],
                             out_path: str = r"./figures/vis_tsne.png",
                             max_per_class: int = 500):
    """
    Generate 2D t-SNE visualizations and small confusion heatmaps for:
      - train (true)
      - test (true)
      - test (predicted by BERT+SVM)
      - test (predicted by BertForSequenceClassification)
    Saves figure to out_path.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    train_titles_s, train_labels_s = _subsample_balanced(train_titles, train_labels, max_per_class)
    test_titles_s, test_labels_s = _subsample_balanced(test_titles, test_labels, max_per_class)

    # get embeddings via bert_svm (uses bert tokenizer + model internals)
    train_emb = bert_svm.get_bert_embeddings(train_titles_s, batch_size=32)
    test_emb = bert_svm.get_bert_embeddings(test_titles_s, batch_size=32)

    # predictions
    svm_preds = None
    if getattr(bert_svm, "svm_classifier", None) is not None:
        svm_preds = bert_svm.svm_classifier.predict(test_emb)

    # ensure bert_clf is loaded (may load from saved dir)
    bert_preds = None
    try:
        if bert_clf.model is None:
            bert_clf.load_model()
        bert_preds = bert_clf.predict(test_titles_s)
    except Exception:
        bert_preds = None

    # t-SNE on combined embeddings for consistent layout
    combined = np.vstack([train_emb, test_emb])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    combined_2d = tsne.fit_transform(combined)

    n_train = train_emb.shape[0]
    train_2d = combined_2d[:n_train]
    test_2d = combined_2d[n_train:]

    cmap = {0: "red", 1: "blue"}
    label_names = {0: "negative", 1: "positive"}

    sns.set(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # Train true labels
    ax = axs[0, 0]
    for cls in np.unique(train_labels_s):
        mask = np.array(train_labels_s) == cls
        ax.scatter(train_2d[mask, 0], train_2d[mask, 1], c=cmap[int(cls)], label=label_names[int(cls)], alpha=0.6, s=10)
    ax.set_title("Train titles (true labels)")
    ax.legend()

    # Test true labels
    ax = axs[0, 1]
    for cls in np.unique(test_labels_s):
        mask = np.array(test_labels_s) == cls
        ax.scatter(test_2d[mask, 0], test_2d[mask, 1], c=cmap[int(cls)], label=label_names[int(cls)], alpha=0.6, s=12)
    ax.set_title("Test titles (true labels)")
    ax.legend()

    # Test predicted by BERT+SVM
    ax = axs[1, 0]
    if svm_preds is not None:
        for cls in np.unique(svm_preds):
            mask = np.array(svm_preds) == cls
            ax.scatter(test_2d[mask, 0], test_2d[mask, 1], c=cmap[int(cls)], label=f"pred {label_names[int(cls)]}", alpha=0.6, s=12, marker='o')
        ax.set_title("Test titles (predictions by BERT+SVM)")
        cm = confusion_matrix(test_labels_s, svm_preds, labels=[0, 1])
        # inset confusion heatmap
        inset_ax = ax.inset_axes([0.65, 0.05, 0.3, 0.3])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=inset_ax)
        inset_ax.set_title("Confusion")
    else:
        ax.text(0.5, 0.5, "SVM preds unavailable", ha='center')
        ax.set_title("BERT+SVM predictions unavailable")
    ax.legend()

    # Test predicted by BertForSequenceClassification
    ax = axs[1, 1]
    if bert_preds is not None:
        for cls in np.unique(bert_preds):
            mask = np.array(bert_preds) == cls
            ax.scatter(test_2d[mask, 0], test_2d[mask, 1], c=cmap[int(cls)], label=f"pred {label_names[int(cls)]}", alpha=0.6, s=12, marker='x')
        ax.set_title("Test titles (predictions by BertForSequenceClassification)")
        cm2 = confusion_matrix(test_labels_s, bert_preds, labels=[0, 1])
        inset_ax2 = ax.inset_axes([0.65, 0.05, 0.3, 0.3])
        sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", ax=inset_ax2)
        inset_ax2.set_title("Confusion")
    else:
        ax.text(0.5, 0.5, "Bert classifier preds unavailable", ha='center')
        ax.set_title("Bert predictions unavailable")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved t-SNE visualization to {out_path}")
    if svm_preds is not None:
        print("Subsampled BERT+SVM accuracy:", accuracy_score(test_labels_s, svm_preds))
    if bert_preds is not None:
        print("Subsampled Bert classifier accuracy:", accuracy_score(test_labels_s, bert_preds))

def main():
    ds = TitleDataset()              # from bert_new
    df = ds.load_data()
    if len(df) == 0:
        print("No training data available in TitleDataset.")
        return

    # split training/validation; use built-in training distribution
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_titles = train_df['title'].tolist()
    train_labels = train_df['label'].tolist()
    val_titles = val_df['title'].tolist()
    val_labels = val_df['label'].tolist()

    # load test file if exists (bert_new has load_test_data), else use validation as test
    try:
        from bert_new import load_test_data
        test_titles, test_labels = load_test_data()
        if len(test_titles) == 0:
            test_titles, test_labels = val_titles, val_labels
    except Exception:
        test_titles, test_labels = val_titles, val_labels

    # initialize classifiers (will try to load saved models if present)
    svm = BertSVMClassifier()
    clf = BertForClassification()

    if not svm.load_model():
        print("SVM model not found/saved. Train or set retrain flag in bert_new before visualization.")
    if not clf.load_model():
        print("Bert classifier model not found/saved. Train or set retrain flag in bert_new before visualization.")

    # run visualization
    visualize_classification(svm, clf, train_titles, train_labels, test_titles, test_labels,
                             out_path=os.path.join(os.path.dirname(__file__), "models", "vis_tsne.png"),
                             max_per_class=500)

if __name__ == "__main__":
    main()