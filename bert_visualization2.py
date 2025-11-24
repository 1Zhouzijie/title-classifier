import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# allow importing bert_new from same folder
sys.path.append(os.path.dirname(__file__))

from bert_new import TitleDataset, BertSVMClassifier  # uses bert_new interfaces

def sample_titles(titles, labels, n=10, random_state=42):
    np.random.seed(random_state)
    titles = np.array(titles)
    labels = np.array(labels)
    if len(titles) <= n:
        return titles.tolist(), labels.tolist()
    # try balanced sampling by class if possible
    unique = np.unique(labels)
    if len(unique) > 1:
        per_class = max(1, n // len(unique))
        idxs = []
        for cls in unique:
            cls_idx = np.where(labels == cls)[0]
            take = min(len(cls_idx), per_class)
            idxs.extend(np.random.choice(cls_idx, take, replace=False).tolist())
        # if not enough, add random
        if len(idxs) < n:
            remaining = list(set(range(len(titles))) - set(idxs))
            idxs.extend(np.random.choice(remaining, n - len(idxs), replace=False).tolist())
        idxs = sorted(idxs)[:n]
    else:
        idxs = sorted(np.random.choice(range(len(titles)), n, replace=False).tolist())
    return titles[idxs].tolist(), labels[idxs].tolist()

def truncate(text, max_len=80):
    t = str(text).replace("\n", " ").strip()
    return (t[:max_len-3] + "...") if len(t) > max_len else t

def plot_tsne_with_labels(embeddings, titles, labels, out_path, title="t-SNE with titles (positive only)"):
    # filter to positive samples only (label == 1)
    labels_arr = np.array(labels)
    pos_mask = labels_arr == 1
    if pos_mask.sum() == 0:
        print("No positive samples to plot.")
        return

    embeddings_pos = embeddings[pos_mask]
    titles_pos = [t for t, m in zip(titles, pos_mask) if m]
    labels_pos = labels_arr[pos_mask]

    # dynamic perplexity: must be < n_samples; fallback to PCA when too few points
    n_samples = embeddings_pos.shape[0]
    if n_samples < 2:
        print("Not enough positive samples for t-SNE/PCA (need >=2).")
        return

    # choose perplexity safely
    perplexity = min(30, max(2, (n_samples - 1) // 3))
    if perplexity >= n_samples:
        perplexity = max(1, n_samples - 1)

    print(f"Running t-SNE with n_samples={n_samples}, perplexity={perplexity} (positive only)")

    try:
        if n_samples <= 2:
            raise ValueError("too few samples for t-SNE, using PCA")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
        emb2 = tsne.fit_transform(embeddings_pos)
    except Exception as e:
        # fallback to PCA for very small sample sizes or if TSNE fails
        from sklearn.decomposition import PCA
        print(f"t-SNE failed ({e}), falling back to PCA.")
        pca = PCA(n_components=2, random_state=42)
        emb2 = pca.fit_transform(embeddings_pos)

    plt.figure(figsize=(10, 8))
    color = "green"
    plt.scatter(emb2[:,0], emb2[:,1], c=color, label="positive (1)", s=80, edgecolors='k', alpha=0.9)

    # annotate each point with truncated title, offset labels to reduce overlap
    for i, txt in enumerate(titles_pos):
        x, y = emb2[i]
        plt.annotate(truncate(txt, 80),
                     xy=(x, y),
                     xytext=(6 + (i % 3) * 2, 6 + (i % 4) * 2),  # small offsets to reduce overlap
                     textcoords='offset points',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.35),
                     arrowprops=dict(arrowstyle="-", alpha=0.2, lw=0.5))

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved labeled t-SNE (positive only) to {out_path}")

def main():
    ds = TitleDataset()
    df = ds.load_data()
    if df is None or len(df) == 0:
        print("No data found in TitleDataset.")
        return

    # split and prepare test pool
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    # try to load separate test set if bert_new provides load_test_data
    try:
        from bert_new import load_test_data
        test_titles, test_labels = load_test_data()
        if len(test_titles) == 0:
            test_titles, test_labels = val_df['title'].tolist(), val_df['label'].tolist()
    except Exception:
        test_titles, test_labels = val_df['title'].tolist(), val_df['label'].tolist()

    # pick small sample ~10 (prefer positives)
    sample_n = 12
    # try to select positive-first sample
    titles_arr = np.array(test_titles)
    labels_arr = np.array(test_labels)
    pos_idx = np.where(labels_arr == 1)[0]
    if len(pos_idx) >= sample_n:
        chosen_idx = np.random.choice(pos_idx, sample_n, replace=False)
    else:
        # take all positives, then fill with random others (but plot will later filter to positives)
        remaining = list(set(range(len(titles_arr))) - set(pos_idx))
        fill = np.random.choice(remaining, max(0, sample_n - len(pos_idx)), replace=False).tolist()
        chosen_idx = np.concatenate([pos_idx, fill]) if len(pos_idx) > 0 else np.random.choice(range(len(titles_arr)), sample_n, replace=False)
    chosen_idx = sorted(chosen_idx)[:sample_n]
    sample_titles_list = titles_arr[chosen_idx].tolist()
    sample_labels_list = labels_arr[chosen_idx].tolist()

    # get embeddings via BertSVMClassifier helper
    svm = BertSVMClassifier()
    # load saved models if present (optional)
    try:
        svm.load_model()
    except Exception:
        pass

    emb = svm.get_bert_embeddings(sample_titles_list, batch_size=8)

    out_file = os.path.join(os.path.dirname(__file__), "figures", "vis_tsne_labeled_positive.png")
    plot_tsne_with_labels(emb, sample_titles_list, sample_labels_list, out_file,
                          title=f"t-SNE of positive titles (sample size={len(sample_titles_list)})")

if __name__ == "__main__":
    main()