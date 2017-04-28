# Kernel trick usage
Generate 2 datasets looking like +- like on attached figures (points coloured differently belong to different classes). 
<p align="center">
  <img src="PCA%231.png"/>
</p>
Apply to them the basic PCA method. Draw diagrams presenting point distribution after this transformation. 
Do the same transformation, but this time using kernel PCA with kernel functions “cosine”, “sigmoid”, and “rbf” 
(in the second and third case play a bit with different values of “coef0” and “gamma” coefficients). Compare the results. 

# Selfies principal components
Treat the photos as a dataset of n*(15+) elements belonging to n classes. Convert them into vectors and apply to them the basic PCA. 
How does the “average face” look like? How does the principal component found by PCA look like (they are very long vectors, 
but we can convert them back into photos)? What is the variance associated to them and how it corresponds to their appearance? 
Reduce the space dimensionality to 5, 15 and 30 dimensions. 
Perform the reverse PCA and check how the original photos look after dimensionality reduction. Finally, 
reduce the dataset to 2 dimensions and plot the elements on 2D surface (colouring them according to the class they belong).
