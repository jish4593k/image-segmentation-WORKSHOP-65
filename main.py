import numpy as np
import os
import cv2
import sys
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time

def recreate_image(centers, labels, rows, cols):
    """
    Recreate the image from the clustered centers and labels.

    Args:
    - centers: Cluster centers
    - labels: Labels assigned to each pixel
    - rows: Number of rows in the image
    - cols: Number of columns in the image

    Returns:
    - image_clusters: Recreated image
    """
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters

if __name__ == '__main__':
    # Ask for segmentation method
    method = input("Which method do you want to choose? Options are: kmeans & gmm ")
    print("The method you selected is ", method, '\n')

    f_dist = np.zeros(10, float)

    for n_cluster in range(10):
        print('Clustering for ', n_cluster + 1, 'clusters\n')
        n_colors = n_cluster + 1

        path = sys.argv[1]
        image_name = sys.argv[2]
        path_file = os.path.join(path, image_name)
        image = cv2.imread(path_file)

        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float64) / 255  # Normalize data from 0 to 1

        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))

        print("Fitting model on a small sub-sample of the data")
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0)[:10000]

        if method == 'gmm':
            model = GMM(n_components=n_colors).fit(image_array_sample)
        else:
            model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

        print("Done in %0.3fs." % (time() - t0))

        
        t0 = time()
        if method == 'gmm':
            print("Predicting color indices on the full image (GMM)")
            labels = model.predict(image_array)
            centers = model.means_
        else:
            print("Predicting color indices on the full image (Kmeans)")
            labels = model.predict(image_array)
            centers = model.cluster_centers_

        print("Done in %0.3fs." % (time() - t0))

   
        dist = np.zeros(n_colors, float)
        for i in range(labels.shape[0]):
            cl = labels[i]
            dist[cl] += np.linalg.norm(image_array[i] - centers[cl])

        f_dist[n_cluster] = np.sum(dist)
        print('The value of the intra-cluster distance is ', f_dist[n_cluster])
        print('\n')

        # Show resulting image
        plt.figure(n_cluster + 1)
        plt.clf()
        plt.axis('off')
        plt.title('Quantized image ({} colors, method={})'.format(n_colors, method))
        plt.imshow(recreate_image(centers, labels, rows, cols))


    plt.figure(0)
    plt.clf()
    plt.axis('off')
    plt.title('Original image')
    plt.imshow(image)

    
    plt.figure(11)
    plt.plot(range(1, 11), f_dist, 'go-', ms=7)
    plt.title('Distance', fontsize=40)
    plt.xlabel('Number of clusters', fontsize=20)
    plt.ylabel('Intra-cluster distance', fontsize=20)
    plt.grid()
    plt.show()
