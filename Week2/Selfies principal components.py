from numpy.random import RandomState
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

image_shape = (64, 64)
rng = RandomState(0)
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

faces2 = np.array(faces[:15])
pca=PCA()
pca.fit(faces2)
averageFace = pca.components_

plt.figure(1)
plt.imshow(averageFace[5].reshape(image_shape), cmap=plt.cm.gray)
plt.title('Average face')


plt.figure(2)
pca4a = PCA(n_components = 5)
faces5D = pca4a.fit_transform(faces)

plt.subplot(2,1,1)
plt.plot(pca4a.explained_variance_, 'ro')
plt.title('Variances for 5D')

pca4b = PCA(n_components = 15)
faces15D = pca4b.fit_transform(faces)

plt.subplot(2,1,2)
plt.plot(pca4b.explained_variance_, 'bo')
plt.title('Variances for 15D')

invFaces5D = pca4a.inverse_transform(faces5D)
invFaces15D = pca4b.inverse_transform(faces15D)

plt.figure(3)
for i in range(15):
    plt.subplot(4,4,i+1)
    plt.imshow(faces[i].reshape(image_shape), cmap=plt.cm.gray)
    plt.axis('off')
plt.suptitle('Faces before applying PCA')

plt.figure(4)
for i in range(15):
    plt.subplot(4,4,i+1)
    plt.imshow(invFaces5D[i].reshape(image_shape), cmap=plt.cm.gray)
    plt.axis('off')
plt.suptitle('Faces 5D after applying PCA and inversing')

plt.figure(5)
for i in range(15):
    plt.subplot(4, 4, i + 1)
    plt.imshow(invFaces15D[i].reshape(image_shape), cmap=plt.cm.gray)
    plt.axis('off')
plt.suptitle('Faces 15D after applying PCA and inversing')


plt.figure(6)
pca6 = PCA(n_components = 2)
faces2D = pca6.fit_transform(faces)
faces2Dx = np.array([faces2D[i][0] for i in range(15)])
faces2Dy = np.array([faces2D[i][0] for i in range(15)])
labels = [i for i in range(15)]
plt.scatter(faces2Dx, faces2Dy, c=labels)
plt.title('Photos on 2D space')

plt.show()
