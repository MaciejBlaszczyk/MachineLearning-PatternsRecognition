import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Generate data
dataLeft = np.array([[np.random.uniform(0, 6), np.random.uniform(0, 10)] for _ in range(200)])
dataRight = np.array([[np.random.uniform(4, 10), np.random.uniform(0, 10)] for _ in range(200)])
dataCircle = np.array([[np.random.uniform(7, 8), np.random.uniform(3, 4)] for _ in range(20)])
data = np.concatenate([dataLeft, dataCircle, dataRight])

# Array of labels
target = [1 if i < 220 else 0 for i in range(420)]

xMin, xMax = data[:, 0].min() - 1, data[:, 0].max() + 1
yMin, yMax = data[:, 1].min() - 1, data[:, 1].max() + 1
xArray, yArray = np.meshgrid(np.arange(xMin, xMax, 0.1), np.arange(yMin, yMax, 0.1))
kernel = ['rbf', 'sigmoid']
C = [4, 10]
gamma = [1, 10]

for ker in kernel:
    for c in C:
        for g in gamma:
            svc = SVC(C = c, gamma = g, kernel = ker)
            svc.fit(data, target)
            result = svc.predict(np.c_[xArray.ravel(), yArray.ravel()])
            result = result.reshape(xArray.shape)
            plt.figure()
            plt.contourf(xArray, yArray, result, alpha=0.5)
            plt.scatter(data[:, 0], data[:, 1], c=target)
            plt.title('Kernel: ' + ker + ' Gamma: ' + str(g) + ' C: ' + str(c))

svc = SVC(kernel = 'linear')
svc.fit(data, target)
result = svc.predict(np.c_[xArray.ravel(), yArray.ravel()])
result = result.reshape(xArray.shape)
plt.figure()
plt.contourf(xArray, yArray, result, alpha=0.5)
plt.scatter(data[:, 0], data[:, 1], c=target)
plt.title('Kernel: linear')

plt.show()