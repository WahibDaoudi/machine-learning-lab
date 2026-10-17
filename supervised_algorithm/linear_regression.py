import numpy as np
from sklearn.datasets import fetch_california_housing


def linear_regression(X, T, alpha=0.01, max_epochs=100000, tolerance=1e-4):
    #Feature scaling (Z-score normalization)
    feature_mean = np.mean(X, axis=0)
    feature_std = np.std(X, axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    X = (X - feature_mean) / feature_std

    #Data prep
    m = X.shape[0]
    one = np.ones((m, 1))
    X = np.hstack((X, one))
    theta = np.zeros((X.shape[1], 1))
    cost_history = []

    #Training loop
    for epoch in range(max_epochs):
        F = np.dot(X, theta)
        e = F - T
        cost = (1 / (2 * m)) * np.sum(e ** 2)
        cost_history.append(cost)

        gradient = (1 / m) * np.dot(X.T, e)

        #Early stopping condition on gradient norm
        if np.linalg.norm(gradient) < tolerance:
            break

        theta = theta - alpha * gradient

    return theta, X, cost_history, epoch + 1, feature_mean, feature_std


if __name__ == "__main__":
    #Extract features and target
    data = fetch_california_housing()
    X = data.data
    T = data.target.reshape(-1, 1)
    theta, X, cost_history, epoch, feature_mean, feature_std = linear_regression(X, T)
    F = np.dot(X, theta)
    print(f"Converged at epoch {epoch}")
    print(f"Training finished. Final cost: {cost_history[-1]:.4f}")
