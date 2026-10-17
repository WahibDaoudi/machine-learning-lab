import numpy as np
from sklearn.datasets import fetch_openml


def fit_perceptron(X_raw, T_raw, max_epochs=100):
    #Extract features , labels , classes
    X = np.hstack((X_raw, np.ones((X_raw.shape[0], 1)))) # -> Append bias column of ones
    classes = np.unique(T_raw)
    T = np.where(T_raw == classes[0], 1.0, -1.0) # -> Convert classes into binary

    #Hyperparameters
    W = np.zeros(X.shape[1]) # -> Initialize weights vector including bias

    #Fit loop
    epoch = 0
    misclass = True
    while misclass and epoch < max_epochs:
        misclass = False
        epoch += 1
        for i in range(X.shape[0]):
            zi = np.dot(W, X[i]) # -> Compute linear output
            if zi * T[i] <= 0:
                W += T[i] * X[i] # -> Update weights on misclassified example
                misclass = True

    return W, classes, epoch


def predict_perceptron(W, classes, user_features):
    #Predictions
    user_input = np.append(user_features, 1.0) # -> Add bias to user input
    z_pred = np.dot(W, user_input)
    return classes[0] if z_pred >= 0 else classes[1]


if __name__ == "__main__":
    import sys

    #User parameters
    dataset_name = sys.argv[1]
    user_features = np.array(sys.argv[2:], dtype=float)

    #Extract features , labels , classes
    data = fetch_openml(dataset_name, version=1, as_frame=False, parser="auto")
    X_raw = data.data.astype(float)
    T_raw = data.target
    W, classes, epoch = fit_perceptron(X_raw, T_raw)
    pred_class = predict_perceptron(W, classes, user_features)

    #Output results
    print("Learned weights (with bias):", W)
    print("Prediction score:", np.dot(W, np.append(user_features, 1.0)))
    print("Predicted class:", pred_class)
