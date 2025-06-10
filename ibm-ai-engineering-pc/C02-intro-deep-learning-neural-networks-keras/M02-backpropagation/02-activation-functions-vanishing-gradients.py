# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Activation Functions and Vanishing Gradients
#

# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
# Sigmoid function and its derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


# %%
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)


# %%
# Generate a range of input values
z = np.linspace(-10, 10, 400)
z

# %%
sigmoid_grad = sigmoid_derivative(z)
sigmoid_grad

# %%
relu_grad = relu_derivative(z)
relu_grad

# %%
# Plot the activation functions
plt.figure(figsize=(12, 6))

# Plot Sigmoid and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z), label='Sigmoid Activation', color='b')
plt.plot(z, sigmoid_grad, label="Sigmoid Derivative", color='r', linestyle='--')
plt.title('Sigmoid Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

# Plot ReLU and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Practice Exercise 1

# %%
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2


# %%
tanh_grad = tanh_derivative(z)
tanh_grad

# %% [markdown]
# ## Practice Exercise 2

# %%
# Generate a range of input values
z = np.linspace(-5, 5, 100)

relu_grad = relu_derivative(z)
tanh_grad = tanh_derivative(z)

# Plot the activation functions
plt.figure(figsize=(12, 6))

# Plot Tanh and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, tanh(z), label='Tanh Activation', color='b')
plt.plot(z, tanh_grad, label="Tanh Derivative", color='r', linestyle='--')
plt.title('Tanh Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

# Plot ReLU and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()
