import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from math import sqrt

# create dataset (2 circles shape)
circlePoints = []
while len(circlePoints) < 100:
    coords = np.random.uniform(2, 6, size = 2)
    if 1 < sqrt((coords[0] - 4) ** 2 + (coords[1] - 4) ** 2) < 2:
        circlePoints.append(coords)
while len(circlePoints) < 300:
    coords = np.random.uniform(0, 8, size = 2)
    if 3 < sqrt((coords[0] - 4) ** 2 + (coords[1] - 4) ** 2) < 4:
        circlePoints.append(coords)
circleLabels = ['b' if i < 100 else 'r' for i in range(300)]
circlePoints = np.array(circlePoints)

# create dataset (line shape)
linePoints = \
    np.array([[x/10, np.random.uniform(0, 2)+x/10] if x < 100 else [(x-100)/10, 12-np.random.uniform(0, 2)-(x-100)/10] for x in range(200)])
lineLabels = ['b' if i < 100 else 'r' for i in range(200)]

# perform PCA with different kernels
kernels = ['linear', 'cosine', 'sigmoid', 'rbf']
for ker in kernels:
    pca = KernelPCA(n_components=2, kernel=ker, coef0=0.02, gamma=3)
    lineResults = pca.fit_transform(linePoints)
    circleResults = pca.fit_transform(circlePoints)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.scatter(circlePoints[:, 0], circlePoints[:, 1], c=circleLabels)
    plt.title('Before using PCA method')
    plt.subplot(2, 2, 2)
    plt.scatter(circleResults[:, 0], circleResults[:, 1], c=circleLabels)
    plt.title('After using PCA method with kernel = ' + str(ker))
    plt.subplot(2, 2, 3)
    plt.scatter(linePoints[:, 0], linePoints[:, 1], c=lineLabels)
    plt.subplot(2, 2, 4)
    plt.scatter(lineResults[:, 0], lineResults[:, 1], c=lineLabels)

plt.show()
