import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./digit-recognizer/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

# Normalize the inputs (scale pixel values from 0-255 to 0-1)
X_train = X_train / 255.
X_dev = X_dev / 255.

def init_params():
    # Slightly deeper network (adding another hidden layer)
    W1 = np.random.randn(128, 784) * np.sqrt(1./784)  # Increased neurons to 128
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(64, 128) * np.sqrt(1./128)   # New middle layer
    b2 = np.zeros((64, 1))
    W3 = np.random.randn(10, 64) * np.sqrt(1./64)     # Output layer
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    A = np.exp(Z)
    return A / np.sum(A, axis=0, keepdims=True)

def forward_props(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def ReLU_deriv(Z):
    return Z > 0

def backward_props(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    beta = 0.9
    
    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)
    vW3 = np.zeros_like(W3)
    vb3 = np.zeros_like(b3)
    
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_props(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_props(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y)
        
        # Update with momentum
        vW1 = beta * vW1 + (1-beta) * dW1
        vb1 = beta * vb1 + (1-beta) * db1
        vW2 = beta * vW2 + (1-beta) * dW2
        vb2 = beta * vb2 + (1-beta) * db2
        vW3 = beta * vW3 + (1-beta) * dW3
        vb3 = beta * vb3 + (1-beta) * db3
        
        # Apply updates
        W1 = W1 - alpha * vW1
        b1 = b1 - alpha * vb1
        W2 = W2 - alpha * vW2
        b2 = b2 - alpha * vb2
        W3 = W3 - alpha * vW3
        b3 = b3 - alpha * vb3
        
        if i % 50 == 0:
            predictions = get_predictions(A3)
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(predictions, Y))
            
    return W1, b1, W2, b2, W3, b3

# Train with slightly lower learning rate due to momentum
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.05, 2000)

