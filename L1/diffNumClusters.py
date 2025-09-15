from sklearn.cluster import KMeans
# from sklearn.cluster import AgglomerativeClustering # Way too much time needed
from scipy.optimize import linear_sum_assignment

from loader import MnistDataloader

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

NUM_CLUSTERS = [10, 15, 18]
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

    print(f"Number of training samples: {len(y_train_flatten)}")

    for k in NUM_CLUSTERS:
        print(f"\n=== KMeans (k={k}) ===")
        model = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
        clusters = model.fit_predict(x_train_flatten)

        # Build class x cluster matrix
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        cluster_sizes = {int(c): int(n) for c, n in zip(unique_clusters, counts)}
        matrix = np.zeros((NUM_CLASSES, len(unique_clusters)), dtype=int)
        for true_label, cluster_id in zip(y_train_flatten, clusters):
            col = np.where(unique_clusters == cluster_id)[0][0]
            matrix[true_label, col] += 1

        '''
        # class-to-cluster accuracy
        row_max = matrix.max(axis=1)
        total_correct = row_max.sum()
        total_samples = matrix.sum()
        accuracy = total_correct / total_samples
        print(f"KMeans - Class-to-cluster accuracy: {accuracy:.4f}")
        '''

        # For each cluster, find dominant class and its % purity
        dominant_class = matrix.argmax(axis=0)
        cluster_to_digit = {int(cl): int(d) for cl, d in zip(unique_clusters, dominant_class)}
        merged_labels = np.array([cluster_to_digit[c] for c in clusters])

        # Group clusters by their assigned digit
        merged_map = {}
        for cl, d in cluster_to_digit.items():
            merged_map.setdefault(d, []).append(int(cl))

        print("\nOriginal clusters merged → digit (with sizes):")
        for d in range(NUM_CLASSES):
            cl_list = merged_map.get(d, [])
            total = sum(cluster_sizes[c] for c in cl_list)
            sizes = ", ".join(f"{c}({cluster_sizes[c]})" for c in cl_list)
            print(f"  Digit {d}: [{sizes}]  total={total}")

        # Now build the merged class x digit matrix (10 x 10)
        merged_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for true_label, merged in zip(y_train_flatten, merged_labels):
            merged_matrix[true_label, merged] += 1

        print(f"Total samples in merged matrix: {merged_matrix.sum()}")

        # Hungarian algorithm to find best 1-to-1 match between digits and clusters
        # Use negative counts as costs (we want to maximize costs)
        cost = -merged_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        ordered_clusters = col_ind.tolist()

        # Reorder columns of matrix
        matrix_reordered = merged_matrix[:, ordered_clusters]

        # Accuracy
        row_max = matrix_reordered.max(axis=1)
        accuracy = row_max.sum() / matrix_reordered.sum()
        print(f"Class-to-cluster accuracy after merge: {accuracy:.4f}")

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_reordered, annot=True, fmt="d", cmap="YlGnBu",
                    xticklabels=[f"Cl {int(unique_clusters[c])}" for c in ordered_clusters],
                    yticklabels=[f"Digit {i}" for i in range(NUM_CLASSES)])
        plt.xlabel("Clusters")
        plt.ylabel("True Digit")
        plt.title(f"KMeans clustering (k={k}) merged -> 10 classes")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run()