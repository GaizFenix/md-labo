from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

from loader import MnistDataloader

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

NUM_CLUSTERS = 10
NUM_CLASSES = 10
SEED = 18


def run():
    input_path = "data"
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

    # Load MINST dataset
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Flatten train data
    x_train_flatten = np.array([np.array(img).flatten() for img in x_train])
    y_train_flatten = np.array(y_train)

    '''
    # Step 1: PCA to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x_train_flatten)

    # Plot the 2D PCA projection colored by true digit label
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train_flatten, cmap='tab10', s=10)
    plt.legend(*scatter.legend_elements(), title="Digits")
    plt.title("MNIST projected to 2D using PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    '''

    methods = {
        "KMeans" : KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init="auto")
        # "Agglomerative" : AgglomerativeClustering(n_clusters=NUM_CLUSTERS),
    }

    for name, model in methods.items():
        print(f"\n=== {name} ===")
        clusters = model.fit_predict(x_train_flatten)

        # Build class x cluster matrix
        unique_clusters = np.unique(clusters)
        matrix = np.zeros((NUM_CLASSES, len(unique_clusters)), dtype=int)
        for true_label, cluster_id in zip(y_train_flatten, clusters):
            col = np.where(unique_clusters == cluster_id)[0][0]
            matrix[true_label, col] += 1
        # row_totals = matrix.sum(axis=1, keepdims=True)
        # matrix_percent = matrix / np.maximum(row_totals, 1) * 100

        # class-to-cluster accuracy
        row_max = matrix.max(axis=1)
        total_correct = row_max.sum()
        total_samples = matrix.sum()
        accuracy = total_correct / total_samples
        print(f"{name} - Class-to-cluster accuracy: {accuracy:.4f}")

        # Hungarian algorithm to find best 1-to-1 match between digits and clusters
        # Use negative counts as costs (we want to maximize costs)
        cost = -matrix
        row_ind, col_ind = linear_sum_assignment(cost)

        # col_ind[i] = best cluster index for digit 1 (row i)
        ordered_clusters = col_ind.tolist()

        # Reorder columns of matrix
        matrix_reordered = matrix[:, ordered_clusters]

        # Percent
        row_totals = matrix_reordered.sum(axis=1, keepdims=True)
        matrix_percent = matrix_reordered / np.maximum(row_totals, 1) * 100

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_percent, annot=True, fmt=".1f", cmap="YlGnBu",
                    xticklabels=[f"Cl {int(unique_clusters[c])}" for c in ordered_clusters],
                    yticklabels=[f"Digit {i}" for i in range(NUM_CLASSES)])
        plt.xlabel("Cluster")
        plt.ylabel("True Digit Class")
        plt.title(f"{name}: Class-to-Cluster (%)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run()