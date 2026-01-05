# PyTorch Basics - Simple Neural Network

This repository is for practicing PyTorch and understanding the basics of neural networks.

## basicNN.ipynb

This notebook demonstrates a simple neural network implemented from scratch using PyTorch.

### Key Concepts Covered:
- **Custom Module Definition**: Creating a neural network class by subclassing `nn.Module`.
- **Parameters**: Defining learnable and frozen parameters (`nn.Parameter`).
- **Forward Pass**: Implementing the forward propagation logic.
- **Activation Functions**: Using ReLU (`F.relu`).
- **Optimization**: Using Stochastic Gradient Descent (`SGD`) to update weights.
- **Training Loop**: Calculating loss, backpropagation, and parameter updates.
- **Visualization**: Plotting the model's output using `matplotlib` and `seaborn`.

### The Model
The model (`BasicNN`) is designed to fit a specific curve (Effectiveness vs. Dose). It has several hardcoded weights and biases, with only one learnable parameter: `final_bias`. The goal of the training loop is to find the optimal value for this bias to minimize the error between the model's predictions and the target labels.
