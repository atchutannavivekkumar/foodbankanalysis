import numpy as np
import matplotlib.pyplot as plt

# Step 1: Dataset Generation
# Parameters for Gaussian distributions
mu1 = [1, 0]
mu2 = [0, 1.5]
cov1 = [[0.9, 0.4], [0.4, 0.9]]
cov2 = [[0.9, 0.4], [0.4, 0.9]]

# Generate 500 samples from each Gaussian distribution
data1 = np.random.multivariate_normal(mu1, cov1, 500)
data2 = np.random.multivariate_normal(mu2, cov2, 500)

# Combine the datasets
X = np.vstack((data1, data2))


# Step 2: Implementing K-Means function
def mykmeans(X, k, c):
    # Initialize centers
    centers = np.array(c)
    num_samples, num_features = X.shape
    max_iters = 10000
    threshold = 0.001
    cluster_labels = np.zeros(num_samples)

    for iteration in range(max_iters):
        # Step 1: Assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        new_labels = np.argmin(distances, axis=1)

        # Step 2: Calculate new centers
        new_centers = np.array(
            [X[new_labels == i].mean(axis=0) if np.any(new_labels == i) else centers[i] for i in range(k)])

        # Step 3: Check for convergence
        if np.linalg.norm(new_centers - centers) <= threshold:
            break
        centers = new_centers
        cluster_labels = new_labels

    return cluster_labels, centers, iteration + 1


# Step 3: Applying K-Means with k = 2
initial_centers_k2 = [(6, 6), (-6, 6)]
labels_k2, centers_k2, iterations_k2 = mykmeans(X, 2, initial_centers_k2)

# Step 4: Applying K-Means with k = 4
initial_centers_k4 = [(6, 6), (-6, -6), (6, -6), (-6, 6)]
labels_k4, centers_k4, iterations_k4 = mykmeans(X, 4, initial_centers_k4)

# Plotting the results for k = 2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels_k2, cmap='viridis', alpha=0.5)
plt.scatter(centers_k2[:, 0], centers_k2[:, 1], c='red', marker='X', s=200, label='Centers')
plt.title(f'K-Means Clustering (k=2)\nFinal Centers: {centers_k2}\nIterations: {iterations_k2}')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Plotting the results for k = 4
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_k4, cmap='viridis', alpha=0.5)
plt.scatter(centers_k4[:, 0], centers_k4[:, 1], c='red', marker='X', s=200, label='Centers')
plt.title(f'K-Means Clustering (k=4)\nFinal Centers: {centers_k4}\nIterations: {iterations_k4}')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

plt.tight_layout()
plt.show()

# Output the final centers and iterations
print(centers_k2, iterations_k2, centers_k4, iterations_k4)
