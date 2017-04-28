import numpy as np
import math
import matplotlib.pyplot as plt

ratio = []
numberOfPoints = 100
numberOfDimensions = 101
hyperCubeEdge = 1

for dimensions in range(2, numberOfDimensions):
    # generate random points in hypercube
    points = np.array([np.random.uniform(0, hyperCubeEdge, size = dimensions) for _ in range(numberOfPoints)])

    # create array for distances between points
    distances = np.empty(int((numberOfPoints * (numberOfPoints - 1)) / 2))

    # calculate distances between points
    temp = 1
    position = 0
    for i in range(numberOfPoints):
        for j in range(temp, numberOfPoints):
            distances[position] = (math.sqrt(math.fsum((points[i] - points[j]) ** 2)))
            position += 1
        temp += 1

    # calculate mean
    mean = math.fsum(distances)/len(distances)

    # calculate standard deviation
    stddev = math.sqrt((math.fsum((distances - mean) ** 2))/len(distances))

    # ratio between stddev and mean
    ratio.append(stddev/mean)

plt.plot([_ for _ in range(2, numberOfDimensions)], ratio, 'bo')
plt.axis([0, 101, 0, 0.5])
plt.ylabel('Ratio')
plt.xlabel('Number Of Dimensions')
plt.title('Ratio (Mean / Standard Deviation) in dependance from amount of dimensions')
plt.show()
