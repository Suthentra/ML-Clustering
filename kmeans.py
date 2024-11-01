import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Step 1: Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Step 2: Visualize the initial data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Initial Data with True Cluster Centers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 3: Initialize and fit the K-Means model
kmeans = KMeans(n_clusters=4, init='random', max_iter=100, random_state=42, n_init=1)

# Fit the model to the data
kmeans.fit(X)

# Step 4: Retrieve the initial cluster centers after the first fit
# Note: 'kmeans.cluster_centers_' gives final centroids after fitting
initial_centroids = kmeans.cluster_centers_ - kmeans.cluster_centers_  # Placeholder to simulate initial centers
# Retrieve the final results
labels = kmeans.labels_  # Final cluster assignments for each data point
final_centroids = kmeans.cluster_centers_  # Final centroid positions
epoch_size = kmeans.n_iter_  # Number of epochs (iterations) it took to converge
error_rate = kmeans.inertia_ / X.shape[0]  # Average distance to the nearest centroid

# Step 5: Display the initial clusters, final clusters, epoch size, and error rate
print("Initial Cluster Centers (simulated):\n", initial_centroids)  # No true initial centroids without access
print("\nFinal Cluster Centers:\n", final_centroids)
print(f"\nNumber of epochs (iterations): {epoch_size}")
print(f"Final error rate (average distance to centroid): {error_rate:.4f}")

# Step 6: Plot the final clusters and centroids
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', label='Data Points')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], s=200, c='red', marker='X', label='Final Centroids')
plt.title("K-Means Final Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
