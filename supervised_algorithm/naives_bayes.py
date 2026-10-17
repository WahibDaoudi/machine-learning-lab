import numpy as np
from sklearn.datasets import fetch_openml


def fit_naive_bayes(X, T):
    #Storage parameters
    classes = np.unique(T)
    means = np.zeros((len(classes), X.shape[1]))
    vars = np.zeros((len(classes), X.shape[1]))
    priors = np.zeros(len(classes))

    #Fit loop -> Calculate means, vars and priors for each class
    for i, c in enumerate(classes):
        X_c = X[T == c]
        means[i] = np.mean(X_c, axis=0)
        vars[i] = np.var(X_c, axis=0) + 1e-9
        priors[i] = len(X_c) / len(T)

    return classes, means, vars, priors


def predict_naive_bayes(classes, means, vars, priors, user_features):
    #Calculate log P(C|X)
    p_c_x = np.zeros(means.shape[0])

    for i in range(means.shape[0]):
        log_p_x_c = 0.0
        for j in range(len(user_features)):
            x = user_features[j]
            mu = means[i, j]
            sigma2 = vars[i, j]
            log_p_x_c += -0.5 * np.log(2 * np.pi * sigma2) - ((x - mu) ** 2) / (2 * sigma2)

        #log(P(C|X)) proportional scores
        p_c_x[i] = np.log(priors[i]) + log_p_x_c

    return classes[np.argmax(p_c_x)]


if __name__ == "__main__":
    import sys

    #User parameters
    dataset_name = sys.argv[1]
    user_features = np.array(sys.argv[2:], dtype=float)

    #Extract features and labels
    data = fetch_openml(dataset_name, version=1, as_frame=False, parser="auto")
    X = data.data.astype(float)
    T = data.target
    classes, means, vars, priors = fit_naive_bayes(X, T)
    label = predict_naive_bayes(classes, means, vars, priors, user_features)

    print(f"Dataset : {dataset_name}\n")
    print(f"Features : {user_features}\n")
    print(f"For these features the prediction of the label is : {label}")
