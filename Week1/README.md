# Curse of Dimensionality 1
We have a hyperball with a radius equal to 1.0 inscribed inside a hypercube with edges length equal to 2.0. Hyperball in multidimensional space is defined as a set of points whose distance from its centre is no greater than its radius. We randomly fill the hypercube with evenly distributed points. What % of those points would be inside the hyperball, and what % outside – in the corners?

# Cursle of Dimensionality 2
We have a hypercube with edges length equal to 1.0. We randomly fill it with evenly distributed points. What is the ratio between the standard deviation of distance between those points and the average distance between them?

# KNN
We perform the task using the legendary (from the pattern recognition point of view) iris dataset (available e.g. in scikit-learn). For the purpose of this assignment you are required to use only the first two dimensions: sepal length and sepal width. You need to implement the k-NN method (the one we talked about in the classroom), Euclidean distance (trivial) and Mahalanobis distance.

The task is to prepare 8 plots showing how given areas of space would be classified by the following variants of k-NN:
* Euclidean distance, k=1;
* Euclidean distance, k=5;
* Mahalanobis distance, k=1;
* Mahalanobis distance, k=5;
