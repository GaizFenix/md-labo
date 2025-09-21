from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

from loader import MnistDataloader

from os.path import join
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns

NUM_CLUSTERS = [10, 15, 18, 50, 100, 200]
NUM_CLASSES = 10
SEED = 18


def build_train_cluster_mapping(y_train, clusters_train):
    unique_clusters = np.unique(clusters_train)
    cl2col = {int(c): j for j, c in enumerate(unique_clusters)}

    # train class x cluster
    M = np.zeros((NUM_CLASSES, len(unique_clusters)), dtype=int)
    for t, c in zip(y_train, clusters_train):
        M[t, cl2col[int(c)]] += 1

    # Hungarian: rows=digits, cols=clusters (select 10 clusters)
    cost = -M
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    # Assigned by Hungarian
    for digit, col in zip(row_ind, col_ind):
        mapping[int(unique_clusters[col])] = int(digit)

    # Any unassigned clusters → dominant digit on train
    dominant_digits = M.argmax(axis=0)
    for j, cl in enumerate(unique_clusters):
        cl_int = int(cl)
        if cl_int not in mapping:
            mapping[cl_int] = int(dominant_digits[j])

    return mapping, M, unique_clusters


def build_digit_confusion(y_true, pred_digits):
    """Return 10x10 confusion (rows=true digit, cols=predicted digit)."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for yt, yp in zip(y_true, pred_digits):
        cm[int(yt), int(yp)] += 1
    return cm


def internal_metrics(X, labels, sample_sil=10000):
    """
    Cohesion/separability metrics (no external labels):
      - Silhouette (subsampled for speed)
      - Calinski–Harabasz
      - Davies–Bouldin
    """
    n = len(labels)
    # Silhouette (subsampled)
    sil = silhouette_score(X, labels, sample_size=min(sample_sil, n), random_state=SEED)
    # CH / DB on full set
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return sil, ch, db


def test_plot(k, X_test2d, y_test, pred_digits_test, cm_test):
    """2-panel figure for the TEST set: left scatter (color=true, number=pred), right confusion matrix."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_test), size=min(800, len(y_test)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap('tab10')

    # Left: scatter (color = true digit; number = predicted digit)
    ax = axes[0]
    ax.scatter(X_test2d[idx, 0], X_test2d[idx, 1], c=y_test[idx], s=50, cmap=cmap)
    for i in idx:
        ax.text(X_test2d[i, 0] + 0.4, X_test2d[i, 1] + 0.4, str(int(pred_digits_test[i])),
                fontsize=8, ha='left', va='bottom', color='black',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    ax.set_title(f"TEST scatter (k={k})\ncolor = true digit; number = predicted")
    ax.set_xlabel("PC1 (train-fit)"); ax.set_ylabel("PC2 (train-fit)")

    # Right: confusion matrix (test)
    ax = axes[1]
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=[f"D{i}" for i in range(NUM_CLASSES)],
                yticklabels=[f"D{i}" for i in range(NUM_CLASSES)], ax=ax)
    ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit")
    ax.set_title("Test Confusion Matrix (digits)")

    plt.tight_layout()
    plt.show()


def run():
    input_path = "data"
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

    # Load MNIST
    mnist = MnistDataloader(training_images_filepath, training_labels_filepath,
                            test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten
    X_train = np.array([np.array(img).flatten() for img in x_train], dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_test  = np.array([np.array(img).flatten() for img in x_test ], dtype=np.float32)
    y_test  = np.array(y_test , dtype=np.int32)

    print(f"Number of training samples: {len(y_train)}")
    print(f"Number of test samples:     {len(y_test)}")

    # PCA(2) for visualization ONLY — fit on TRAIN, transform TEST
    pca2 = PCA(n_components=2, random_state=SEED).fit(X_train)
    X_test2d = pca2.transform(X_test)

    # For the final 2x3 grid
    grid_data = [] # list of (k, cm_test)

    # Table header
    print("\n=== Cohesion/Separability metrics (higher Sil/CH better, lower DB better) ===")
    print(f"{'k':>3} | {'Silhouette(train)':>18} {'CH(train)':>12} {'DB(train)':>12} || {'Silhouette(test)':>17} {'CH(test)':>12} {'DB(test)':>12}")
    print("-"*86)

    for k in NUM_CLUSTERS:
        print(f"\n=== KMeans (k={k}) ===")
        km = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
        clusters_train = km.fit_predict(X_train)
        clusters_test  = km.predict(X_test)

        # Metrics (train/test)
        sil_tr, ch_tr, db_tr = internal_metrics(X_train, clusters_train, sample_sil=10000)
        sil_te, ch_te, db_te = internal_metrics(X_test,  clusters_test,  sample_sil=10000)
        print(f"{k:>3} | {sil_tr:18.4f} {ch_tr:12.1f} {db_tr:12.4f} || {sil_te:17.4f} {ch_te:12.1f} {db_te:12.4f}")

        # Build train-based cluster→digit mapping (no leakage)
        cluster_to_digit, train_M, uniq = build_train_cluster_mapping(y_train, clusters_train)

        # Predicted digits on TEST using train mapping
        pred_digits_test = np.array([cluster_to_digit[int(c)] for c in clusters_test], dtype=np.int32)

        # Test confusion matrix (digits vs predicted digits) — ONLY for visualization/comparison
        cm_test = build_digit_confusion(y_test, pred_digits_test)
        grid_data.append((k, cm_test))

        # Plots for TEST
        test_plot(k, X_test2d, y_test, pred_digits_test, cm_test)

    # --------------------------
    # Final 2×3 grid of ConfMats
    # --------------------------
    if grid_data:
        # Consistent color scale across all heatmaps
        vmax = max(cm.max() for _, cm in grid_data)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        for i, (k, cm) in enumerate(grid_data):
            ax = axes[i]
            sns.heatmap(cm, ax=ax, annot=True, fmt="d", cmap="YlGnBu",
                        vmin=0, vmax=vmax, cbar=False,
                        xticklabels=[f"D{i}" for i in range(NUM_CLASSES)],
                        yticklabels=[f"D{i}" for i in range(NUM_CLASSES)])
            ax.set_title(f"k = {k}")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")

        # Hide unused axes if <6 k-values
        for j in range(len(grid_data), 6):
            fig.delaxes(axes[j])

        fig.suptitle("Test Confusion Matrices for Different k", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run()