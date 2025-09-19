import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from os.path import join

from loader import MnistDataloader

SEED = 18
NUM_CLASSES = 10

# Hungarian alignment + confusion matrix
def class_cluster_eval(y_true, clusters, num_classes=NUM_CLASSES):
    mat = np.zeros((num_classes, num_classes), dtype=int)  # rows=true digits, cols=cluster ids
    for t, c in zip(y_true, clusters):
        mat[t, c] += 1
    cost = -mat
    _, col_ind = linear_sum_assignment(cost)
    cm = mat[:, col_ind]
    cluster_to_digit = {int(c): int(i) for i, c in enumerate(col_ind)}
    acc = cm.trace() / cm.sum()
    return cm, acc, cluster_to_digit

def plot_dual(title, X2d, y_true, pred_digits, confmat, classified_n):
    rng = np.random.default_rng(SEED)
    n_points = min(800, len(y_true))
    idx = rng.choice(len(y_true), size=n_points, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cmap = plt.get_cmap('tab10')

    # Left: scatter (color=true digit, number=predicted)
    ax = axes[0]
    sc = ax.scatter(X2d[idx, 0], X2d[idx, 1], c=y_true[idx], s=60, cmap=cmap)
    for i in idx:
        ax.text(X2d[i, 0] + 0.5, X2d[i, 1] + 0.5, str(int(pred_digits[i])),
                fontsize=9, ha='left', va='bottom', color='black',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Legend: True digit colors
    handles = [plt.Line2D([0],[0], marker='o', color='w',
                           markerfacecolor=cmap(d), markersize=6, label=f"D{d+1}")
               for d in range(NUM_CLASSES)]
    ax.legend(handles=handles, title="True digit (color)", loc='center left', bbox_to_anchor=(1, 0.5))

    # Title (two lines) + classified note
    ax.set_title(f"{title}\ncolor = true digit; number = predicted\nclassified: {classified_n} samples")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Right: Confusion matrix
    ax = axes[1]
    x_ticks = [f"C{i+1}" for i in range(NUM_CLASSES)]
    y_ticks = [f"D{i+1}" for i in range(NUM_CLASSES)]
    sns.heatmap(confmat, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=x_ticks, yticklabels=y_ticks, ax=ax)
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Digits")
    ax.set_title("Class vs Cluster")

    plt.tight_layout()
    plt.show()

def main():
    input_path = "data"
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

    loader = MnistDataloader(training_images_filepath, training_labels_filepath,
                             test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = loader.load_data()

    X_train = np.array([np.array(img).flatten() for img in x_train], dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_test  = np.array([np.array(img).flatten() for img in x_test ], dtype=np.float32)
    y_test  = np.array(y_test , dtype=np.int32)

    # Shared 2D projection for visualization
    X_all = np.vstack([X_train, X_test])
    X_2d  = PCA(n_components=2, random_state=SEED).fit_transform(X_all)

    results = []

    # -------------------------
    # A) Train on train, predict on test
    # -------------------------
    km_train = KMeans(n_clusters=NUM_CLASSES, random_state=SEED, n_init="auto")
    km_train.fit(X_train)
    clusters_test = km_train.predict(X_test)
    cm_A, acc_A, map_A = class_cluster_eval(y_test, clusters_test)
    pred_A = np.vectorize(lambda c: map_A[c])(clusters_test)

    classified_A = len(y_test)
    print(f"[Train→Test] Classified: {classified_A} samples")
    plot_dual("KMeans trained on train, predicted on test",
              X_2d[len(y_train):], y_test, pred_A, cm_A, classified_A)
    results.append(("Train→Test", acc_A, classified_A))

    # -------------------------
    # B) Train on all, predict on all
    # -------------------------
    km_all = KMeans(n_clusters=NUM_CLASSES, random_state=SEED, n_init="auto")
    clusters_all = km_all.fit_predict(X_all)
    y_all = np.concatenate([y_train, y_test])
    cm_B, acc_B, map_B = class_cluster_eval(y_all, clusters_all)
    pred_B = np.vectorize(lambda c: map_B[c])(clusters_all)

    classified_B = len(y_all)
    print(f"[All data]   Classified: {classified_B} samples")
    plot_dual("KMeans trained on all data, predicted on all",
              X_2d, y_all, pred_B, cm_B, classified_B)
    results.append(("All data", acc_B, classified_B))

    # -------------------------
    # Print comparison table
    # -------------------------
    print("\n=== KMeans comparison ===")
    print(f"{'Setting':<15} {'Accuracy':<10} {'Classified':<10}")
    print("-"*40)
    for name, acc, n in results:
        print(f"{name:<15} {acc:<10.4f} {n:<10d}")

if __name__ == "__main__":
    main()