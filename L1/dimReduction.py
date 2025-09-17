from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment

from loader import MnistDataloader

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

NUM_CLASSES = 10
SEED = 18


def class_cluster_eval(y_true, cluster_ids, num_classes=NUM_CLASSES):
    # Build matrix (true digit x cluster)
    mat = np.zeros((num_classes, num_classes), dtype=int)
    for t, c in zip(y_true, cluster_ids):
        mat[t, c] += 1

    # Costs for Hungarian algorithm
    cost = -mat
    _, col_ind = linear_sum_assignment(cost)

    # Reorder columns by Hungarian alignment
    cm = mat[:, col_ind]

    # cluster_id -> predicted digit index (column in cm)
    cluster_to_digit = {int(col_id): int(j) for j, col_id in enumerate(col_ind)}

    # Compute accuracy
    acc = cm.trace() / cm.sum()
    return cm, acc, cluster_to_digit

def plot_case(title, X_plot2d, y_true, pred_digits, conf_mat):
    """
    Two-panel figure:
      left: 200-point scatter (color = predicted digit, text = true digit)
      right: Confusion matrix heatmap (counts, integers)
    """
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_true), size=200, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: scatter with labels
    ax = axes[0]
    sc = ax.scatter(X_plot2d[idx, 0], X_plot2d[idx, 1], c=pred_digits[idx], s=20, cmap='tab10')
    for i in idx:
        ax.text(X_plot2d[i, 0], X_plot2d[i, 1], str(int(y_true[i])), fontsize=7,
                ha='center', va='center')
    ax.set_title(f"{title} — sample of 200\n(color = predicted digit; text = true digit)")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    # Right panel: confusion matrix
    ax = axes[1]
    sns.heatmap(conf_mat, ax=ax, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=[f"D{i}" for i in range(NUM_CLASSES)],
                yticklabels=[f"D{i}" for i in range(NUM_CLASSES)])
    ax.set_xlabel("Predicted (after alignment)")
    ax.set_ylabel("True digit")
    ax.set_title("Class → Cluster (counts)")

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

    # Flatten train data
    X = np.array([np.array(img).flatten() for img in x_train], dtype=np.float32)
    y = np.array(y_train, dtype=np.int32)
    n = len(y)
    print(f"Training samples: {n}")

    results = []

    # -------- Case A: No DR (cluster in pixel space); plot with PCA(2) --------
    kmeans_raw = KMeans(n_clusters=NUM_CLASSES, random_state=SEED, n_init="auto")
    clusters_raw = kmeans_raw.fit_predict(X)
    cm_raw, acc_raw, cl2dig_raw = class_cluster_eval(y, clusters_raw)
    pred_digits_raw = np.vectorize(lambda c: cl2dig_raw[c])(clusters_raw)  # color by predicted digit
    # 2D plot coords for visualization only
    X_plot_raw = PCA(n_components=2, random_state=SEED).fit_transform(X)
    plot_case("KMeans on raw pixels", X_plot_raw, y, pred_digits_raw, cm_raw)
    results.append(("Raw", acc_raw))

    # -------- Case B: PCA → KMeans (use 50 PCs), plot with same 2D PCA --------
    X_pca50 = PCA(n_components=100, random_state=SEED).fit_transform(X)
    kmeans_pca = KMeans(n_clusters=NUM_CLASSES, random_state=SEED, n_init="auto")
    clusters_pca = kmeans_pca.fit_predict(X_pca50)
    cm_pca, acc_pca, cl2dig_pca = class_cluster_eval(y, clusters_pca)
    pred_digits_pca = np.vectorize(lambda c: cl2dig_pca[c])(clusters_pca)
    # For visualization, use PCA(2) on the 50D space (cleaner than pixel PCA(2))
    X_plot_pca2 = PCA(n_components=2, random_state=SEED).fit_transform(X_pca50)
    plot_case("PCA(50) → KMeans", X_plot_pca2, y, pred_digits_pca, cm_pca)
    results.append(("PCA(50)", acc_pca))

    # t-SNE was way too slow on this dataset, and required too much computational power

    # -------- Accuracy table --------
    print("\nClass-to-cluster accuracy (aligned via Hungarian):")
    print("-----------------------------------------------")
    for name, acc in results:
        print(f"{name:<10}  {acc:.4f}")


if __name__ == "__main__":
    run()