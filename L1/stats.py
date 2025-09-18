import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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
    print("\n=== Instances by label (train) ===")
    counts_train = Counter(y_train)
    counts = [counts_train[d] for d in range(10)]
    for d in range(10):
        print(f"Digit {d}: {counts[d]}")

    # Bar plot with counts on top
    plt.figure(figsize=(10,4))
    bars = plt.bar(range(10), counts, color='skyblue')
    plt.xticks(range(10))
    plt.title("Training set: instances by label")
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.ylim(0, max(counts)*1.15) # add some headroom on top

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, count + 200, str(count),
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # --- 3) 2D PCA scatter plot ---
    print("\nBuilding PCA(2) projection for visualization...")
    X_pca2 = PCA(n_components=2, random_state=SEED).fit_transform(X_train)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_train), size=3000, replace=False)

    fig, ax = plt.subplots(figsize=(8,6))
    cmap = plt.get_cmap('tab10')
    sc = ax.scatter(X_pca2[idx, 0], X_pca2[idx, 1], c=y_train[idx], s=25, cmap=cmap)

    # Legend: color → class
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(d), markersize=6, label=f"Digit {d}")
        for d in range(10)
    ]
    ax.legend(handles=handles, title="True digit", loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_title("MNIST — PCA(2) projection (3000 sample points)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    explore_mnist()
