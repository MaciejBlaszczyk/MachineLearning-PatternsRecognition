from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generating data set (points)
points = np.array([[1, 1]])
i = 1
while i < 7:
    j = 1
    while j < 7:
        temp = np.array([[np.random.uniform(i, i+1), np.random.uniform(j, j+1)] for _ in range(50)])
        points = np.concatenate((points, temp), axis = 0)
        j = j + 2
    i = i + 2

x = KMeans(n_clusters=9,n_init=1).fit(points)

# calculating silhouette score for kmeans++
kMeansPP = KMeans(n_clusters=9, init='k-means++', n_init=1)
kMeansPPScore = []
for _ in range(10):
    kMeansPPLabels = kMeansPP.fit_predict(points)
    kMeansPPScore.append(silhouette_score(points, kMeansPPLabels))

# calculating silhouette score for kmeans with random init
kMeansRandom = KMeans(n_clusters=9, init='random', n_init=1)
kMeansRandomScore = []
for _ in range(10):
    kMeansRandomLabels = kMeansRandom.fit_predict(points)
    kMeansRandomScore.append(silhouette_score(points, kMeansRandomLabels))

# calculating silhouette score for kmeans with forgy init
kMeansForgyScore = []
for _ in range(10):
    forgyIndexes = np.random.uniform(0, 450, size=9)
    forgyPoints = np.array([points[int(forgyIndexes[i])] for i in range(len(forgyIndexes))])
    kMeansForgy = KMeans(n_clusters=9, init=forgyPoints, n_init=1)
    kMeansForgyLabels = kMeansForgy.fit_predict(points)
    kMeansForgyScore.append(silhouette_score(points, kMeansForgyLabels))

# calculating silhouette score for kmeans with random partition init
kMeansRPScore = []
for _ in range(10):
    temp = np.random.uniform(0, 9, size=450)
    RPIndexes = np.array([int(temp[i]) for i in range(len(temp))])

    results = [[[], []] for _ in range(9)]
    for cluster, coords in zip(RPIndexes, points):
        results[cluster][0].append(coords[0])
        results[cluster][1].append(coords[1])
    RPPoints = np.array([[sum(results[i][0])/len(results[i][0]), sum(results[i][1])/len(results[i][1])] for i in range(9)])

    kMeansRP = KMeans(n_clusters=9, init=RPPoints, n_init=1)
    kMeansRPLabels = kMeansRP.fit_predict(points)
    kMeansRPScore.append(silhouette_score(points, kMeansRPLabels))


plt.subplot(2, 2, 1)
plt.errorbar(range(10), kMeansPPScore, yerr = np.std(kMeansPPScore), fmt ='-o')
plt.axis([-1, 10, 0, 1])
plt.title('Silhouette Score (higher = better) for kMeans++')
plt.grid()

plt.subplot(2, 2, 2)
plt.errorbar(range(10), kMeansRandomScore, yerr = np.std(kMeansRandomScore), fmt='-o')
plt.axis([-1, 10, 0, 1])
plt.title('Silhouette Score (higher = better) for kMeans with random init')
plt.grid()

plt.subplot(2, 2, 3)
plt.errorbar(range(10), kMeansForgyScore, yerr = np.std(kMeansForgyScore), fmt='-o')
plt.axis([-1, 10, 0, 1])
plt.title('Silhouette Score (higher = better) for kMeans with forgy init')
plt.grid()

plt.subplot(2, 2, 4)
plt.errorbar(range(10), kMeansRPScore, yerr = np.std(kMeansRPScore), fmt='-o')
plt.axis([-1, 10, 0, 1])
plt.title('Silhouette Score (higher = better) for kMeans with random partial init')
plt.grid()

plt.show()
