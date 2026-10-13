import sys
import numpy as np
from sklearn.datasets import fetch_openml

#User parameters
dataset_name = sys.argv[1]
user_features = np.array(sys.argv[2:], dtype=float)

#Extract features , labels , classes
data = fetch_openml(dataset_name, version=1, as_frame=False, parser="auto")
X_raw = data.data.astype(float)
X = np.hstack((X_raw, np.ones((X_raw.shape[0], 1)))) # -> Append bias column of ones
T_raw = data.target
classes = np.unique(T_raw)
T = np.where(T_raw == classes[0], 1.0, -1.0) # -> Convert classes into binary 

#Hyperparameters
W = np.zeros(X.shape[1]) # -> Initialize weights vector including bias 
max_epochs = 100 # -> Prevent infinite loop if data is not linearly separable

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

#Predictions
user_input = np.append(user_features, 1.0) # -> Add bias to input features
z_pred = np.dot(W, user_input)
pred_class = classes[0] if z_pred >= 0 else classes[1]

#Output results
print("Learned weights (with bias):", W)
print("Prediction score:", z_pred)
print("Predicted class:", pred_class)