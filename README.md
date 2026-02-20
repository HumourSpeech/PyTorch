# PyTorch Learning Journey

This repository contains notebooks and scripts for learning PyTorch, starting from basic tensor operations to building and training neural networks.

## Course Structure

### 1. Tensor Basics [`tensor_basics/`](tensor_basics/tensor_operations.ipynb)
*   **`tensor_operations.ipynb`**: Introduction to PyTorch tensors. Covers:
    *   Tensor creation and initialization.
    *   Basic mathematical operations.
    *   Reshaping and slicing.
    *   Numpy to Tensor conversion.

### 2. Autograd [`02_autograd/`](02_autograd/gradients.ipynb)
*   **`gradients.ipynb`**: Understanding PyTorch's automatic differentiation engine (`autograd`).
    *   Calculating gradients.
    *   `requires_grad` and backpropagation.

### 3. Neural Networks [`03_neural_networks/`](03_neural_networks/simple_nn.ipynb)
*   **`simple_nn.ipynb`**: Building neural network components.
    *   Defining layers using `torch.nn`.
    *   Activation functions.

### 4. Training Loop [`04_training_loop/`](04_training_loop/train.ipynb)
*   **`train.ipynb`**: Putting it all together.
    *   Forward pass.
    *   Loss calculation.
    *   Backward pass.
    *   Optimizer step.

### 5. Datasets & DataLoaders [`05_Datasets_DataLoaders/`](05_Datasets_DataLoaders)
*   **`datasets.ipynb`**: Working with PyTorch Datasets.
    *   Custom Dataset classes (`torch.utils.data.Dataset`).
    *   Data preprocessing and transforms.
*   **`dataloaders.ipynb`**: Using DataLoaders for efficient training.
    *   Batching, shuffling, and loading data (`torch.utils.data.DataLoader`).
    *   Iterating through batches.

## Projects / Examples

### Basic Neural Network [`basicNN.ipynb`](basicNN.ipynb)

This notebook demonstrates a simple neural network implemented from scratch using PyTorch to fit a curve.

**Key Concepts Covered:**
- **Custom Module Definition**: Creating a neural network class by subclassing `nn.Module`.
- **Parameters**: Defining learnable and frozen parameters (`nn.Parameter`).
- **Forward Pass**: Implementing the forward propagation logic.
- **Activation Functions**: Using ReLU (`F.relu`).
- **Optimization**: Using Stochastic Gradient Descent (`SGD`) to update weights.
- **Training Loop**: Calculating loss, backpropagation, and parameter updates.
- **Visualization**: Plotting the model's output using `matplotlib` and `seaborn`.

**The Model:**
The model (`BasicNN`) is designed to fit a specific curve (Effectiveness vs. Dose). It has several hardcoded weights and biases, with only one learnable parameter: `final_bias`. The goal of the training loop is to find the optimal value for this bias to minimize the error between the model's predictions and the target labels.
