# Generate 2 datasets looking like +- like on attached figures (points coloured differently belong to different classes)
# Apply to them the basic PCA method. Draw diagrams presenting point distribution after this transformation.
# Do the same transformation, but this time using kernel PCA with kernel functions “cosine”, “sigmoid”, and “rbf”
# (in the second and third case play a bit with different values of “coef0” and “gamma” coefficients).
# Compare the results.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from math import sqrt


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


linePoints = \
    np.array([[x/10, np.random.uniform(0, 2)+x/10] if x < 100 else [(x-100)/10, 12-np.random.uniform(0, 2)-(x-100)/10] for x in range(200)])
lineLabels = ['b' if i < 100 else 'r' for i in range(200)]


pca = PCA(n_components = 2)
pca.fit(linePoints)
lineResults = pca.components_
pca.fit(circlePoints)
circleResults = pca.components_
print(lineResults)

plt.figure(1)
plt.subplot(2, 2, 1)
plt.scatter(circlePoints[:, 0], circlePoints[:, 1], c = circleLabels)
plt.axis([-10, 10, -10, 10])
plt.grid()
plt.title('Before using PCA method')
plt.subplot(2, 2, 2)
plt.scatter(circleResults[:, 0], circleResults[:, 1], c = circleLabels)
plt.axis([-10, 10, -10, 10])
plt.grid()
plt.title('After using PCA method')
plt.subplot(2, 2, 3)
plt.scatter(linePoints[:, 0], linePoints[:, 1], c = lineLabels)
plt.axis([-15, 15, -15, 15])
plt.grid()
plt.subplot(2, 2, 4)
plt.scatter(lineResults[:, 0], lineResults[:, 1], c = lineLabels)
plt.axis([-15, 15, -15, 15])
plt.grid()

plt.show()
'''
kpcaCosine = KernelPCA(n_components = 2, kernel = 'cosine')
lineResultsKernelCosine = kpcaCosine.fit_transform(linePoints)
circleResultsKernelCosine = kpcaCosine.fit_transform(circlePoints)


plt.figure(2)
plt.subplot(2, 2, 1)
plt.scatter(circlePoints[:, 0], circlePoints[:, 1], c = circleLabels)
plt.axis([-10, 10, -10, 10])
plt.grid()
plt.title('Before using KPCA method (cosine)')
plt.subplot(2, 2, 2)
plt.scatter(circleResultsKernelCosine[:, 0], circleResultsKernelCosine[:, 1], c = circleLabels)
plt.axis([-1, 1, -1, 1])
plt.grid()
plt.title('After using KPCA method (cosine)')
plt.subplot(2, 2, 3)
plt.scatter(linePoints[:, 0], linePoints[:, 1], c = lineLabels)
plt.axis([-15, 15, -15, 15])
plt.grid()
plt.subplot(2, 2, 4)
plt.scatter(lineResultsKernelCosine[:, 0], lineResultsKernelCosine[:, 1], c = lineLabels)
plt.axis([-1, 1, -1, 1])
plt.grid()


kpcaSigmoid = KernelPCA(n_components = 2, kernel = 'sigmoid', coef0 = 0.02)
lineResultsKernelSigmoid = kpcaSigmoid.fit_transform(linePoints)
circleResultsKernelSigmoid = kpcaSigmoid.fit_transform(circlePoints)

plt.figure(3)
plt.subplot(2, 2, 1)
plt.scatter(circlePoints[:, 0], circlePoints[:, 1], c = circleLabels)
plt.axis([-10, 10, -10, 10])
plt.grid()
plt.title('Before using KPCA method (sigmoid)')
plt.subplot(2, 2, 2)
plt.scatter(circleResultsKernelSigmoid[:, 0], circleResultsKernelSigmoid[:, 1], c = circleLabels)
plt.axis([-1, 1, -1, 1])
plt.grid()
plt.title('After using KPCA method (sigmoid)')
plt.subplot(2, 2, 3)
plt.scatter(linePoints[:, 0], linePoints[:, 1], c = lineLabels)
plt.axis([-15, 15, -15, 15])
plt.grid()
plt.subplot(2, 2, 4)
plt.scatter(lineResultsKernelSigmoid[:, 0], lineResultsKernelSigmoid[:, 1], c = lineLabels)
plt.axis([-1, 1, -1, 1])
plt.grid()


kpcaRbf = KernelPCA(n_components = 2, kernel = 'rbf', gamma = 3)
lineResultsKernelRBF = kpcaRbf.fit_transform(linePoints)
circleResultsKernelRBF = kpcaRbf.fit_transform(circlePoints)


plt.figure(4)
plt.subplot(2, 2, 1)
plt.scatter(circlePoints[:, 0], circlePoints[:, 1], c = circleLabels)
plt.axis([-10, 10, -10, 10])
plt.grid()
plt.title('Before using KPCA method (rbf)')
plt.subplot(2, 2, 2)
plt.scatter(circleResultsKernelRBF[:, 0], circleResultsKernelRBF[:, 1], c = circleLabels)
plt.axis([-1, 1, -1, 1])
plt.grid()
plt.title('After using KPCA method (rbf)')
plt.subplot(2, 2, 3)
plt.scatter(linePoints[:, 0], linePoints[:, 1], c = lineLabels)
plt.axis([-15, 15, -15, 15])
plt.grid()
plt.subplot(2, 2, 4)
plt.scatter(lineResultsKernelRBF[:, 0], lineResultsKernelRBF[:, 1], c = lineLabels)
plt.axis([-1, 1, -1, 1])
plt.grid()
plt.show()
'''