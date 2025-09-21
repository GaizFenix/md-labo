from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from loader import MnistDataloader

from os.path import join
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns

NUM_CLUSTERS = 10
NUM_CLASSES = 10
SEED = 18

# ---------- Utilities ----------

def build_train_mapping(y_train, clusters_train, num_classes=NUM_CLASSES):
    """
    Build cluster_id -> digit mapping using TRAIN ONLY.
    Returns:
      mapping: dict cluster_id -> digit
      cm_train: 10x10 confusion (true digit x mapped prediction) on TRAIN
      acc_train: diagonal / total on TRAIN
    """
    uniq = np.unique(clusters_train)
    cl2col = {int(c): j for j, c in enumerate(uniq)}
    M = np.zeros((num_classes, len(uniq)), dtype=int)  # rows=true digit, cols=cluster id (ordered by uniq)
    for t, c in zip(y_train, clusters_train):
        M[int(t), cl2col[int(c)]] += 1

    # Hungarian to pick best 1-to-1 assignment for 10 digits
    cost = -M
    row_ind, col_ind = linear_sum_assignment(cost)

    # Start with Hungarian mapping (selected clusters)
    mapping = {int(uniq[c]): int(d) for d, c in zip(row_ind, col_ind)}

    # Any unselected clusters -> dominant digit on TRAIN
    dom_digits = M.argmax(axis=0)
    for j, cl in enumerate(uniq):
        cl = int(cl)
        if cl not in mapping:
            mapping[cl] = int(dom_digits[j])

    # Build 10x10 confusion on TRAIN using mapping
    pred_train_digits = np.array([mapping[int(c)] for c in clusters_train], dtype=np.int32)
    cm_train = np.zeros((num_classes, num_classes), dtype=int)
    for yt, yp in zip(y_train, pred_train_digits):
        cm_train[int(yt), int(yp)] += 1
    acc_train = np.trace(cm_train) / np.sum(cm_train)
    return mapping, cm_train, acc_train

def class_confusion(y_true, pred_digits, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for yt, yp in zip(y_true, pred_digits):
        cm[int(yt), int(yp)] += 1
    acc = np.trace(cm) / np.sum(cm)
    return cm, acc

def internal_metrics(X, labels, sample_sil=10000):
    n = len(labels)
    sil = silhouette_score(X, labels, sample_size=min(sample_sil, n), random_state=SEED)
    ch  = calinski_harabasz_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    return sil, ch, db

def plot_test_case(title, X_plot2d_test, y_test, pred_digits_test, cm_test):
    """
    Two-panel TEST figure:
      left: scatter (color = true digit; number = predicted digit)
      right: Confusion matrix (digits)
    """
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_test), size=min(800, len(y_test)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap('tab10')

    # Left: scatter
    ax = axes[0]
    ax.scatter(X_plot2d_test[idx, 0], X_plot2d_test[idx, 1], c=y_test[idx], s=60, cmap=cmap)
    for i in idx:
        ax.text(X_plot2d_test[i, 0] + 0.5, X_plot2d_test[i, 1] + 0.5, str(int(pred_digits_test[i])),
                fontsize=9, ha='left', va='bottom', color='black',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    # Legend (true class colors)
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=cmap(d), markersize=6, label=f"D{d}")
               for d in range(NUM_CLASSES)]
    ax.legend(handles=handles, title="True digit (color)", loc="lower right", frameon=True, framealpha=0.9)
    ax.set_title(f"{title}\n(color = true digit; number = assigned digit)")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

    # Right: confusion matrix
    ax = axes[1]
    sns.heatmap(cm_test, ax=ax, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=[f"D{i}" for i in range(NUM_CLASSES)],
                yticklabels=[f"D{i}" for i in range(NUM_CLASSES)])
    ax.set_xlabel("Assigned (after train mapping)")
    ax.set_ylabel("True digit")
    ax.set_title("Test: Class → Assigned (counts)")

    plt.tight_layout()
    plt.show()

# ---------- Main ----------

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

    # Flatten (float32)
    X_train = np.array([np.array(img).flatten() for img in x_train], dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_test  = np.array([np.array(img).flatten() for img in x_test ], dtype=np.float32)
    y_test  = np.array(y_test , dtype=np.int32)

    print(f"Training samples: {len(y_train)}, features {X_train.shape[1]}")

    results = []  # (name, acc_train, acc_test, sil_tr, ch_tr, db_tr, sil_te, ch_te, db_te)

    # ---- Case A: Raw pixels ----
    km = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init="auto")
    cl_tr = km.fit_predict(X_train)
    cl_te = km.predict(X_test)

    # Train-only mapping; evaluate train & test accuracies (for context)
    mapping, cm_tr_digits, acc_tr = build_train_mapping(y_train, cl_tr)
    pred_te_digits = np.array([mapping[int(c)] for c in cl_te], dtype=np.int32)
    cm_te_digits, acc_te = class_confusion(y_test, pred_te_digits)

    # Internal cohesion/separability metrics
    sil_tr, ch_tr, db_tr = internal_metrics(X_train, cl_tr, sample_sil=10000)
    sil_te, ch_te, db_te = internal_metrics(X_test,  cl_te, sample_sil=10000)
    results.append((f"Raw({X_train.shape[1]})", acc_tr, acc_te, sil_tr, ch_tr, db_tr, sil_te, ch_te, db_te))

    # Plot TEST (2D PCA for visualization only)
    pca2_raw = PCA(n_components=2, random_state=SEED).fit(X_train)
    X_test2d_raw = pca2_raw.transform(X_test)
    plot_test_case("KMeans on raw pixels (TEST)", X_test2d_raw, y_test, pred_te_digits, cm_te_digits)

    # ---- PCA variants: 50, 100, 200, 500 ----
    for p in [50, 100, 200, 500]:
        pca_p = PCA(n_components=p, random_state=SEED).fit(X_train)
        Xtr_p = pca_p.transform(X_train)
        Xte_p = pca_p.transform(X_test)

        km = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init="auto")
        cl_tr = km.fit_predict(Xtr_p)
        cl_te = km.predict(Xte_p)

        mapping, cm_tr_digits, acc_tr = build_train_mapping(y_train, cl_tr)
        pred_te_digits = np.array([mapping[int(c)] for c in cl_te], dtype=np.int32)
        cm_te_digits, acc_te = class_confusion(y_test, pred_te_digits)

        sil_tr, ch_tr, db_tr = internal_metrics(Xtr_p, cl_tr, sample_sil=10000)
        sil_te, ch_te, db_te = internal_metrics(Xte_p, cl_te, sample_sil=10000)
        results.append((f"PCA({p})", acc_tr, acc_te, sil_tr, ch_tr, db_tr, sil_te, ch_te, db_te))

        # 2D viz space (fit on train in this p-dim space)
        pca2 = PCA(n_components=2, random_state=SEED).fit(Xtr_p)
        X_test2d = pca2.transform(Xte_p)
        plot_test_case(f"PCA({p}) → KMeans (TEST)", X_test2d, y_test, pred_te_digits, cm_te_digits)

    # ---- Metrics table ----
    print("\nCohesion/Separability & Mapping Accuracies (train mapping applied)")
    print(f"{'Variant':<12} {'Acc_tr':>7} {'Acc_te':>7} || {'Sil_tr':>7} {'CH_tr':>10} {'DB_tr':>7} || {'Sil_te':>7} {'CH_te':>10} {'DB_te':>7}")
    print("-"*100)
    for name, a_tr, a_te, s_tr, c_tr, d_tr, s_te, c_te, d_te in results:
        print(f"{name:<12} {a_tr:7.4f} {a_te:7.4f} || {s_tr:7.4f} {c_tr:10.1f} {d_tr:7.4f} || {s_te:7.4f} {c_te:10.1f} {d_te:7.4f}")


if __name__ == "__main__":
    run()