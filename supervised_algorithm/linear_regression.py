import numpy as np 
from sklearn.datasets import fetch_california_housing

# Extract features and target
data = fetch_california_housing()
X = data.data
T = data.target.reshape(-1, 1)

# Feature scaling (Z-score normalization)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Data prep
m = X.shape[0]
one = np.ones((m, 1))
X = np.hstack((X, one))
theta = np.zeros((X.shape[1], 1))
cost_history = []

# Hyperparameters
alpha = 0.01
max_epochs = 100000
tolerance = 1e-4

# Training loop
for epoch in range(max_epochs):
    F = np.dot(X, theta)
    e = F - T
    cost = (1 / (2 * m)) * np.sum(e ** 2)
    cost_history.append(cost)

    gradient = (1 / m) * np.dot(X.T, e)
    
    # Early stopping condition on gradient norm
    if np.linalg.norm(gradient) < tolerance:
        print(f"Converged at epoch {epoch}")
        break

    theta = theta - alpha * gradient

print(f"Training finished. Final cost: {cost_history[-1]:.4f}")