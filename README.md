# MNIST Digit Recognition Neural Network

A deep neural network implementation from scratch for handwritten digit recognition, achieving 95.24% accuracy on the MNIST dataset.

## 🎯 Overview

This project demonstrates how to build and train a neural network without using any deep learning frameworks. It recognizes handwritten digits (0-9) by learning from the MNIST dataset, which contains 70,000 grayscale images of handwritten digits.

## 🚀 Quick Start

1. Clone this repository
2. Ensure you have the required dependencies:
   ```bash
   pip install numpy pandas matplotlib
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## 🧮 Network Architecture

### Layer Structure

Input Layer (784 neurons) → Hidden Layer 1 (128 neurons) → Hidden Layer 2 (64 neurons) → Output Layer (10 neurons)

### Details
- **Input Layer**: 784 neurons (28x28 pixel images flattened)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one for each digit)

## 🔧 Technical Implementation

### Data Preprocessing
- Images are normalized from 0-255 to 0-1 scale
- Labels are converted to one-hot encoded vectors
- Data is split into training and development sets

### Key Components

#### 1. Weight Initialization


W1 = np.random.randn(128, 784) np.sqrt(1./784)
b1 = np.zeros((128, 1))

... similar for W2, b2, W3, b3

#### `forward_props()`
Performs forward propagation through the network:

Z1 = W1.dot(X) + b1
A1 = ReLU(Z1)

... similar for other layers

#### `backward_props()`
Computes gradients using backpropagation:

dZ3 = A3 - one_hot_Y
dW3 = 1/m dZ3.dot(A2.T)
# ... similar for other layers

#### `gradient_descent()`
Updates parameters using momentum optimization:

vW1 = beta vW1 + (1-beta) dW1
W1 = W1 - alpha vW1
# ... similar for other parameters

## 🔍 Mathematical Foundation

### Forward Pass
1. Z₁ = W₁X + b₁
2. A₁ = ReLU(Z₁)
3. Z₂ = W₂A₁ + b₂
4. A₂ = ReLU(Z₂)
5. Z₃ = W₃A₂ + b₃
6. A₃ = Softmax(Z₃)

### Backward Pass
1. dZ₃ = A₃ - Y
2. dW₃ = 1/m * dZ₃A₂ᵀ
3. db₃ = 1/m * Σ(dZ₃)
4. Similar computations for other layers

## 🛠 Possible Improvements

1. Add Convolutional Layers
   - Would better capture spatial relationships
   - Could improve accuracy to ~99%

2. Implement Regularization
   - L2 regularization
   - Dropout layers
   - Batch normalization

3. Modern Optimizers
   - Adam optimizer
   - RMSprop
   - AdaGrad

4. Data Augmentation
   - Random rotations
   - Small shifts
   - Elastic deformations

## 📚 References

1. MNIST Dataset: http://yann.lecun.com/exdb/mnist/
2. Deep Learning Book (Goodfellow et al.)
3. Neural Networks and Deep Learning (Michael Nielsen)

## 👥 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

