# 1Treat the photos as a dataset of n*(15+) elements belonging to n classes.
# 2Convert them into vectors and apply to them the basic PCA. How does the “average face” look like?
# 3How does the principal component found by PCA look like (they are very long vectors, but we can convert them back into photos)?
# 4What is the variance associated to them and how it corresponds to their appearance? Reduce the space dimensionality to 5, 15 and 30 dimensions.
# 5Perform the reverse PCA and check how the original photos look after dimensionality reduction.
# 6Finally, reduce the dataset to 2 dimensions and plot the elements on 2D surface (colouring them according to the class they belong). Are they easily separable?


from numpy.random import RandomState
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

from sklearn.decomposition import PCA

image_shape = (64, 64)
rng = RandomState(0)
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data


faces2 = np.array([faces[i] for i in range(15)])
pca=PCA()
pca.fit(faces2)
averageFace = pca.components_
print(averageFace)
plt.figure(1)
plt.imshow(averageFace[5].reshape(image_shape), cmap=plt.cm.gray)
plt.title('Average face')


'''4444444444444444444444444444444444444444'''

plt.figure(2)
pca4a = PCA(n_components = 5)
faces5D = pca4a.fit_transform(faces2)

plt.subplot(2,1,1)
plt.plot(pca4a.explained_variance_, 'ro')
plt.title('Variances for 5D')

pca4b = PCA(n_components = 15)
faces15D = pca4b.fit_transform(faces2)

plt.subplot(2,1,2)
plt.plot(pca4b.explained_variance_, 'bo')
plt.title('Variances for 15D')

'''55555555555555555555555555555555555555555'''

invFaces5D = pca4a.inverse_transform(faces5D)
invFaces15D = pca4b.inverse_transform(faces15D)

plt.figure(3)
for i in range(15):
    plt.subplot(4,4,i+1)
    plt.imshow(faces2[i].reshape(image_shape), cmap=plt.cm.gray)
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

'''66666666666666666666666666666666666666666'''

plt.figure(6)
pca6 = PCA(n_components = 2)
faces2D = pca6.fit_transform(faces2)
#print(faces2D)
faces2Dx = np.array([faces2D[i][0] for i in range(15)])
faces2Dy = np.array([faces2D[i][0] for i in range(15)])
labels = [i for i in range(15)]
plt.scatter(faces2Dx, faces2Dy, c=labels)
plt.title('Photos on 2D space')




plt.show()
