from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from loader import MnistDataloader

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

NUM_CLUSTERS = 40
NUM_CLASSES = 10

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

    # Step 2: Try clustering to see how many groups form
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=18)
    clusters = kmeans.fit_predict(x_train_flatten)

    '''
    # Plot clusters without using labels (unsupervised view)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', s=10)
    plt.title("KMeans Clusters (k=10) on PCA-reduced MNIST")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    '''

    # Class-to-cluster evaluation
    matrix = np.zeros((NUM_CLASSES, NUM_CLUSTERS), dtype=int)

    for true_label, cluster_id in zip(y_train_flatten, clusters):
        matrix[true_label, cluster_id] += 1

    # Normalize rows to percentages (optional)
    matrix_percent = matrix / matrix.sum(axis=1, keepdims=True) * 100

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_percent, annot=True, fmt=".1f", cmap="YlGnBu",
                xticklabels=[f"Cluster {i}" for i in range(NUM_CLUSTERS)],
                yticklabels=[f"Digit {i}" for i in range(NUM_CLASSES)])
    plt.xlabel("Cluster")
    plt.ylabel("True Digit Class")
    plt.title("Class-to-Cluster Assignment (%)")
    plt.tight_layout()
    plt.show()

    '''
    # Helper function to show a list of images with their relating titles
    def show_images(images, title_texts):
        cols = 5
        rows = int(len(images)/cols) + 1
        plt.figure(figsize=(30,20))
        index = 1    
        for x in zip(images, title_texts):        
            image = x[0]        
            title_text = x[1]
            plt.subplot(rows, cols, index)        
            plt.imshow(image, cmap=plt.cm.gray)
            if (title_text != ''):
                plt.title(title_text, fontsize = 15);        
            index += 1
        plt.tight_layout()
        plt.show()
    '''

    '''
    # Show some random training and test images 
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])        
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

    show_images(images_2_show, titles_2_show)
    '''

if __name__ == "__main__":
    run()