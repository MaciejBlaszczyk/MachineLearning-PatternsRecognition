# Benchmarking the initialisation method
Generate a dataset similar to the one presented on the enclosed picture. 
<p align="center">
  <img src="kmeans.png" width="350"/>
</p>
Run on it a k-means clustering algorithm (with k=9) and the following initialisation methods:
- fully random;
- Forgy;
- random partition;
- k-means++.
Measure a chosen clustering quality metric (either Davies-Bouldin index, Dunn index, or Silhouette) after each algorithm iteration. 
Present the results on a plot (remember to repeat experiment multiple times and show the standard deviation of the values as errorbars). 
