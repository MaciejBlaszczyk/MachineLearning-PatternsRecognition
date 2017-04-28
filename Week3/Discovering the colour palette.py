import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle


def imageShow():
    labelID = 0
    image = np.zeros((w, h, d))
    for i in range(w):
        for j in range(h):
            image[i][j] = clusterCenters[labels[labelID]]
            labelID += 1
    return image


china = load_sample_image("china.jpg")
w, h, d = original_shape = china.shape
image_array = np.reshape(china, (w * h, d))
image_array = image_array / 255
plt.figure()
plt.imshow(china)

colorsNumber = [2, 5, 10, 20, 50]
for color in colorsNumber:
    centroids = shuffle(image_array, random_state=0)[:color]
    kmeans = KMeans(n_clusters=color, init=centroids, n_init=1).fit(shuffle(image_array, random_state=0))
    labels = kmeans.predict(image_array)
    clusterCenters = kmeans.cluster_centers_
    image = imageShow()
    plt.figure()
    plt.imshow(image)

plt.show()