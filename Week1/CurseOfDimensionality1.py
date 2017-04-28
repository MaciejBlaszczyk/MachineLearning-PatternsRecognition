import numpy as np
import math
import matplotlib.pyplot as plt

numberOfPoints = 1000
numberOfDimensions = 21
hyperCubeEdge = 2
hyperBallRadius = 1

percentResults = []


for dimensions in range(2, numberOfDimensions):
    # generate random points in hypercube
    points = np.array([np.random.uniform(-1, 1, size = dimensions) for _ in range(numberOfPoints)])

    # squared points to calculate distances
    squaredPoints = points ** 2

    # calculate distances between points
    distances = np.array([math.sqrt(math.fsum(squaredPoints[i])) for i in range(numberOfPoints)])

    # find points which are inside hyperball
    results = np.array([1 if distances[i] < 1 else 0 for i in range(numberOfPoints)])

    percent = (math.fsum(results)/numberOfPoints)*100
    print("Dimensions:", dimensions, " ", percent, "% points in hyperball")
    percentResults.append(percent)


plt.plot([_ for _ in range(2, numberOfDimensions)], percentResults)
plt.axis([0, 21, -10, 110])
plt.xlabel('Dimensions')
plt.ylabel('% of points in hypersphere')
plt.title('% of points in hypersphere in dependance from amount of dimensions')
plt.show()
