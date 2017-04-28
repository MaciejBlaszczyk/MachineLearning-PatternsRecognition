from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

iris = datasets.load_iris()
points = np.array(iris.data[:, :2])
types = iris.target

# prepare arrays
EucK1 = np.array(types)
EucK5 = np.array(types)
ManK1 = np.array(types)
ManK5 = np.array(types)

# calculate distances between points (euclidean metric) and exchange zeros with nans
distancesEuc = np.array(distance.cdist(points, points, 'euclidean'))
distancesEuc[distancesEuc == 0] = np.nan

# calculate distances between points (manhattan metric) and exchange zeros with nans
distancesMan = np.array(distance.cdist(points, points, 'cityblock'))
distancesMan[distancesMan == 0] = np.nan

# perform classification (K = 1)
for i in range(len(points)):
    minIndex = np.nanargmin(distancesEuc[i])
    EucK1[i] = types[minIndex]
    minIndex = np.nanargmin(distancesMan[i])
    ManK1[i] = types[minIndex]

# perform classification (K = 5)
for i in range(len(points)):
    tempResults = []
    for _ in range(5):
        minIndex = np.nanargmin(distancesEuc[i])
        distancesEuc[i][minIndex] = np.nan
        tempResults.append(types[minIndex])
    if tempResults.count(0) > 2:
        EucK5[i] = 0
    elif tempResults.count(1) > 2:
        EucK5[i] = 1
    elif tempResults.count(2) > 2:
        EucK5[i] = 2
    else:
        EucK5[i] = 3

    tempResults = []
    for _ in range(5):
        minIndex = np.nanargmin(distancesMan[i])
        distancesMan[i][minIndex] = np.nan
        tempResults.append(types[minIndex])
    if tempResults.count(0) > 2:
        ManK5[i] = 0
    elif tempResults.count(1) > 2:
        ManK5[i] = 1
    elif tempResults.count(2) > 2:
        ManK5[i] = 2
    else:
        ManK5[i] = 3

plt.figure(1)
plt.scatter(points[:, 0], points[:, 1], c=types)
plt.title('Data Set')
plt.figure(2)
plt.scatter(points[:, 0], points[:, 1], c=EucK1)
plt.title('Euclidean 1 NN')
plt.figure(3)
plt.scatter(points[:, 0], points[:, 1], c=ManK1)
plt.title('Manhattan 1 NN')
plt.figure(4)
plt.scatter(points[:, 0], points[:, 1], c=EucK5)
plt.title("Euclidean 5 NN")
plt.figure(5)
plt.scatter(points[:, 0], points[:, 1], c=ManK5)
plt.title('Manhattan 5 NN')

plt.show()
