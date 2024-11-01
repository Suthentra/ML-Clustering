import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Step 1: Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Step 2: Visualize the initial data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Initial Data with True Cluster Centers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 3: Initialize and fit the Agglomerative Clustering model
agglo = AgglomerativeClustering(n_clusters=4)

# Fit the model to the data
labels = agglo.fit_predict(X)

# Step 4: Retrieve the final results
# Note: Agglomerative Clustering does not explicitly compute centroids
final_centroids = []  # Placeholder for centroids
for cluster in range(agglo.n_clusters):
    cluster_points = X[labels == cluster]
    if len(cluster_points) > 0:
        centroid = np.mean(cluster_points, axis=0)
        final_centroids.append(centroid)
final_centroids = np.array(final_centroids)

# Calculate the silhouette score (a measure of how similar an object is to its own cluster compared to other clusters)
silhouette_avg = silhouette_score(X, labels)

# Step 5: Display the initial clusters, final clusters, epoch size, and error rate
print("\nInitial Cluster Labels (based on true clusters):\n", y_true)
print("\nFinal Cluster Labels:\n", labels)
print("\nFinal Cluster Centers:\n", final_centroids)
print(f"\nNumber of merge operations (similar to epochs): {agglo.n_clusters - 1}")  # Number of merges
print(f"Final silhouette score (error rate approximation): {silhouette_avg:.4f}")

# Step 6: Plot the final clusters and centroids
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', label='Data Points')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], s=200, c='red', marker='X', label='Final Centroids')
plt.title("Agglomerative Clustering Final Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
