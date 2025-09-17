from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment

from loader import MnistDataloader

from os.path import join
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns

NUM_CLUSTERS = 10
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

    acc = cm.trace() / cm.sum()
    return cm, acc, cluster_to_digit

def plot_case(title, X_plot2d, y_true, pred_digits, conf_mat):
    """
    Two-panel figure:
      left: n-point scatter (color = predicted digit, text = true digit)
      right: Confusion matrix heatmap (counts, integers)
    """
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_true), size=300, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: scatter with labels
    ax = axes[0]
    cmap = plt.get_cmap('tab10')
    sc = ax.scatter(X_plot2d[idx, 0], X_plot2d[idx, 1], c=y_true[idx], s=80, cmap=cmap)
    # Draw the true digit as text (bigger, outlined for readability)
    for i in idx:
        ax.text(
            X_plot2d[i, 0] + 0.5, X_plot2d[i, 1] + 0.5, str(int(pred_digits[i])), 
            fontsize=11, ha='left', va='bottom', color='black',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
            )
    ax.set_title(f"{title} — sample of 300\n(color = true digit; text = assigned digit)")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    # Color info box (predicted digit → color)
    handles = []
    for d in range(NUM_CLASSES):
        handles.append(plt.Line2D([0], [0], marker='o', linestyle='',
                                  markersize=6, markerfacecolor=cmap(d), markeredgecolor='none',
                                  label=f"D{d}"))
    leg = ax.legend(handles=handles, title="True digit (color)", loc="lower right",
                    frameon=True, framealpha=0.9)
    leg._legend_box.align = "left"

    # Right panel: confusion matrix
    ax = axes[1]
    sns.heatmap(conf_mat, ax=ax, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=[f"C{i}" for i in range(NUM_CLUSTERS)],
                yticklabels=[f"D{i}" for i in range(NUM_CLASSES)])
    ax.set_xlabel("Clustered/assigned (after alignment)")
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
    n,d = X.shape
    print(f"Training samples: {n}, features {d}")

    results = []

    # -------- Case A: Raw pixels (cluster in raw space); plot with PCA(2) of raw --------
    kmeans_raw = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init="auto")
    clusters_raw = kmeans_raw.fit_predict(X)
    cm_raw, acc_raw, cl2dig_raw = class_cluster_eval(y, clusters_raw)
    pred_digits_raw = np.vectorize(lambda c: cl2dig_raw[c])(clusters_raw)  # color by predicted digit
    # 2D plot coords for visualization only
    X_plot_raw = PCA(n_components=2, random_state=SEED).fit_transform(X)
    plot_case("KMeans on raw pixels", X_plot_raw, y, pred_digits_raw, cm_raw)
    results.append((f"Raw({d})", acc_raw))

    # -------- PCA variants: 50, 100, 200, 500 components ----------
    for p in [50, 100, 200, 500]:
        X_pca = PCA(n_components=p, random_state=SEED).fit_transform(X)
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init="auto")
        clusters = kmeans.fit_predict(X_pca)
        cm, acc, cl2dig = class_cluster_eval(y, clusters)
        pred_digits = np.vectorize(lambda c: cl2dig[c])(clusters)
        # 2D visualization from this PCA space
        X_plot_pca2 = PCA(n_components=2, random_state=SEED).fit_transform(X_pca)
        plot_case(f"PCA({p}) → KMeans", X_plot_pca2, y, pred_digits, cm)
        results.append((f"PCA({p})", acc))

    # t-SNE was way too slow on this dataset, and required too much computational power

    # -------- Accuracy table --------
    print("\nClass-to-cluster accuracy (aligned via Hungarian):")
    print("-----------------------------------------------")
    for name, acc in results:
        print(f"{name:<10}  {acc:.4f}")


if __name__ == "__main__":
    run()