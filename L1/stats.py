import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
from os.path import join

from loader import MnistDataloader

SEED = 18

def explore_mnist():
    input_path = "data"
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

    # Load MNIST data
    loader = MnistDataloader(training_images_filepath, training_labels_filepath,
                             test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = loader.load_data()

    # Flatten and convert to arrays
    X_train = np.array([np.array(img).flatten() for img in x_train], dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_test = np.array([np.array(img).flatten() for img in x_test], dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    # --- 1) Basic info ---
    print("=== Basic Dataset Info ===")
    print(f"Train samples: {len(y_train)}  |  Test samples: {len(y_test)}")
    print(f"Image size: {X_train.shape[1]} pixels per image (flattened)")

    # --- 2) Instances by label ---
    print("\n=== Instances by label ===")
    counts_train = Counter(y_train)
    counts_test = Counter(y_test)
    counts_all = {d: counts_train[d] + counts_test[d] for d in range(10)}

    print("Train:")
    for d in range(10):
        print(f"Digit {d}: {counts_train[d]}")
    print("\nTest:")
    for d in range(10):
        print(f"Digit {d}: {counts_test[d]}")

    # Bar plot (stacked) with counts on top
    train_vals = np.array([counts_train[d] for d in range(10)])
    test_vals = np.array([counts_test[d] for d in range(10)])
    total_vals = train_vals + test_vals

    fig, ax = plt.subplots(figsize=(10,4))
    bars_train = ax.bar(range(10), train_vals, color='skyblue', label='Train')
    bars_test = ax.bar(range(10), test_vals, bottom=train_vals, color='orange', label='Test')

    for i, (t, te) in enumerate(zip(train_vals, test_vals)):
        ax.text(i, t/2, str(t), ha='center', va='center', fontsize=8, color='black')
        ax.text(i, t + te/2, str(te), ha='center', va='center', fontsize=8, color='black')
        ax.text(i, t+te+300, str(t+te), ha='center', va='bottom', fontsize=9, color='black')

    ax.set_xticks(range(10))
    ax.set_title("Instances per label (Train+Test)")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_ylim(0, max(total_vals)*1.2)
    plt.tight_layout()
    plt.show()

    # --- 3) Dual PCA(2) scatter plot ---
    print("\nBuilding PCA(2) projection for visualization...")

    rng = np.random.default_rng(SEED)
    idx_train = rng.choice(len(y_train), size=3000, replace=False)
    idx_test = rng.choice(len(y_test), size=3000, replace=False)

    X_train_2d = PCA(n_components=2, random_state=SEED).fit_transform(X_train)
    X_test_2d = PCA(n_components=2, random_state=SEED).fit_transform(X_test)

    fig, axes = plt.subplots(1, 3, figsize=(15,6), gridspec_kw={'width_ratios':[1,0.2,1]})
    cmap = plt.get_cmap('tab10')

    # Left scatter (train)
    axes[0].scatter(X_train_2d[idx_train, 0], X_train_2d[idx_train, 1],
                    c=y_train[idx_train], s=10, cmap=cmap)
    axes[0].set_title(f"MNIST — PCA(2) (Train, 3000 points)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    # Middle (empty for legend)
    axes[1].axis('off')

    # Right scatter (test)
    axes[2].scatter(X_test_2d[idx_test, 0], X_test_2d[idx_test, 1],
                    c=y_test[idx_test], s=10, cmap=cmap)
    axes[2].set_title(f"MNIST — PCA(2) (Test, 3000 points)")
    axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")

    # Legend centered
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(d), markersize=6, label=f"Digit {d}")
        for d in range(10)
    ]
    axes[1].legend(handles=handles, title="True digit", loc='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    explore_mnist()