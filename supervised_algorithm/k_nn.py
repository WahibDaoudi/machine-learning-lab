import sys
import numpy as np
from sklearn.datasets import fetch_openml

#User parameters
dataset_name = sys.argv[1]
k = int(sys.argv[2])
point = np.array(sys.argv[3:], dtype=float)


# Extract features and labels 
data = data = fetch_openml(dataset_name, version=1, as_frame=False, parser="auto")
X = data.data.astype(float)
T = data.target


def k_nn(point: np.ndarray ,k: int):
    
    # Euclidean distance between all dataset points and the input point
    distances = np.sqrt(np.sum((X - point) ** 2, axis=1))
    sorted_indices = np.argsort(distances)
    
    # Extract the k nearest neighbours' labels directly with slicing and force k to be odd
    if k % 2 == 0 :
        k += 1
    k_labels = T[sorted_indices[0:k]]

    # Return the most frequent label
    values, counts = np.unique(k_labels, return_counts=True)
    return values[np.argmax(counts)]



pred = k_nn(point, k)
print(f"Prediction for dataset '{dataset_name}': {pred}")