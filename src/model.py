"""
model.py
--------
Pure NumPy implementation of a Multiclass Multilayer Perceptron (MLP).

Design choices:
- No deep-learning framework (TensorFlow / PyTorch) — keeps the implementation
  transparent and directly demonstrates the mathematical operations.
- Activation: ReLU for hidden layers, Softmax for output layer.
- Loss: Categorical cross-entropy.
- Training: Mini-batch / Stochastic Gradient Descent (SGD) with optional momentum.
- Weight initialisation: He initialisation (suitable for ReLU activations).
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Activation functions
# ──────────────────────────────────────────────────────────────────────────────

def relu(z):
    """ReLU: f(z) = max(0, z)"""
    return np.maximum(0.0, z)


def relu_derivative(z):
    """Derivative of ReLU: 1 if z > 0 else 0"""
    return (z > 0).astype(float)


def softmax(z):
    """
    Numerically stable row-wise Softmax.
    softmax(z_i) = exp(z_i - max(z)) / sum(exp(z_j - max(z)))
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ──────────────────────────────────────────────────────────────────────────────
# Loss function
# ──────────────────────────────────────────────────────────────────────────────

def cross_entropy_loss(y_pred, y_true_onehot, epsilon=1e-12):
    """
    Categorical cross-entropy loss.

    L = - (1/N) * sum_i sum_k y_true[i,k] * log(y_pred[i,k])

    Parameters
    ----------
    y_pred       : (N, K) predicted probabilities from softmax
    y_true_onehot: (N, K) one-hot encoded true labels
    """
    N = y_pred.shape[0]
    log_probs = np.log(np.clip(y_pred, epsilon, 1.0))
    loss = -np.sum(y_true_onehot * log_probs) / N
    return loss


def one_hot(y, num_classes):
    """Converts integer label array to one-hot matrix."""
    N = len(y)
    oh = np.zeros((N, num_classes))
    oh[np.arange(N), y] = 1.0
    return oh


# ──────────────────────────────────────────────────────────────────────────────
# MLP
# ──────────────────────────────────────────────────────────────────────────────

class MLP:
    """
    Multilayer Perceptron for multiclass classification.

    Architecture (example with 2 hidden layers):
        Input (d)  →  Dense + ReLU  →  Dense + ReLU  →  Dense + Softmax  →  Output (K)

    Parameters
    ----------
    layer_sizes : list[int]
        Full layer configuration including input and output dimensions.
        e.g. [32, 64, 32, 3] means input_dim=32, hidden=[64,32], output_dim=3.
    learning_rate : float
        SGD step size.
    momentum : float
        SGD momentum coefficient (0 = vanilla SGD, 0.9 = standard momentum).
    random_state : int
        RNG seed for reproducible weight initialisation.
    """

    def __init__(self, layer_sizes, learning_rate=0.01, momentum=0.0,
                 random_state=42):
        self.layer_sizes   = layer_sizes
        self.lr            = learning_rate
        self.momentum      = momentum
        self.rng           = np.random.default_rng(random_state)
        self.n_layers      = len(layer_sizes) - 1   # number of weight matrices
        self.num_classes   = layer_sizes[-1]

        self._init_weights()
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }

    # ── Weight initialisation (He) ────────────────────────────────────────────

    def _init_weights(self):
        self.W = []   # weight matrices
        self.b = []   # bias vectors
        self.vW = []  # momentum terms for W
        self.vb = []  # momentum terms for b

        for i in range(self.n_layers):
            fan_in  = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # He initialisation: std = sqrt(2 / fan_in)
            std = np.sqrt(2.0 / fan_in)
            W_i = self.rng.normal(0.0, std, size=(fan_in, fan_out))
            b_i = np.zeros((1, fan_out))
            self.W.append(W_i)
            self.b.append(b_i)
            self.vW.append(np.zeros_like(W_i))
            self.vb.append(np.zeros_like(b_i))

    # ── Forward propagation ───────────────────────────────────────────────────

    def forward(self, X):
        """
        Forward pass through the network.

        For each hidden layer l:
            Z[l] = A[l-1] @ W[l] + b[l]
            A[l] = ReLU(Z[l])

        For the output layer:
            Z[L] = A[L-1] @ W[L] + b[L]
            A[L] = Softmax(Z[L])

        Returns cached activations and pre-activations for backprop.
        """
        self.Z = []   # pre-activations
        self.A = [X]  # activations; A[0] = input

        for i in range(self.n_layers):
            z = self.A[i] @ self.W[i] + self.b[i]
            self.Z.append(z)
            if i < self.n_layers - 1:
                a = relu(z)         # hidden layers
            else:
                a = softmax(z)      # output layer
            self.A.append(a)

        return self.A[-1]   # predicted probabilities, shape (N, K)

    # ── Backward propagation ──────────────────────────────────────────────────

    def backward(self, y_onehot):
        """
        Backpropagation using chain rule.

        Output layer gradient (softmax + cross-entropy combined):
            dL/dZ[L] = (A[L] - y_onehot) / N          (δ[L])

        Hidden layer gradient:
            dL/dZ[l] = (δ[l+1] @ W[l+1].T) * ReLU'(Z[l])    (δ[l])

        Weight gradients:
            dL/dW[l] = A[l-1].T @ δ[l]
            dL/db[l] = mean(δ[l], axis=0)
        """
        N = self.A[0].shape[0]
        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers

        # Output layer delta
        delta = (self.A[-1] - y_onehot) / N

        for i in reversed(range(self.n_layers)):
            grads_W[i] = self.A[i].T @ delta
            grads_b[i] = np.mean(delta, axis=0, keepdims=True)
            if i > 0:
                # Propagate gradient to previous layer
                delta = (delta @ self.W[i].T) * relu_derivative(self.Z[i - 1])

        return grads_W, grads_b

    # ── SGD weight update ─────────────────────────────────────────────────────

    def update_weights(self, grads_W, grads_b):
        """
        SGD with momentum update:
            v[t+1] = momentum * v[t] + lr * grad
            W[t+1] = W[t] - v[t+1]
        """
        for i in range(self.n_layers):
            self.vW[i] = self.momentum * self.vW[i] + self.lr * grads_W[i]
            self.vb[i] = self.momentum * self.vb[i] + self.lr * grads_b[i]
            self.W[i] -= self.vW[i]
            self.b[i] -= self.vb[i]

    # ── Prediction helpers ────────────────────────────────────────────────────

    def predict_proba(self, X):
        """Returns softmax probability matrix (N, K)."""
        return self.forward(X)

    def predict(self, X):
        """Returns integer class predictions (N,)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y):
        """Computes classification accuracy."""
        return float(np.mean(self.predict(X) == y))

    def loss(self, X, y_int):
        """Computes cross-entropy loss."""
        proba = self.predict_proba(X)
        y_oh  = one_hot(y_int, self.num_classes)
        return cross_entropy_loss(proba, y_oh)

    # ── Architecture summary ──────────────────────────────────────────────────

    def summary(self):
        """Prints a human-readable architecture summary."""
        print("=" * 55)
        print("  MLP Architecture Summary")
        print("=" * 55)
        total_params = 0
        for i in range(self.n_layers):
            n_w = self.W[i].size
            n_b = self.b[i].size
            params = n_w + n_b
            total_params += params
            act = "ReLU" if i < self.n_layers - 1 else "Softmax"
            print(f"  Layer {i+1}: ({self.layer_sizes[i]:4d} → {self.layer_sizes[i+1]:4d})"
                  f"  {act:8s}  params={params:6d}")
        print("-" * 55)
        print(f"  Total trainable parameters: {total_params:,}")
        print("=" * 55)
        return total_params

    def get_config(self):
        return {
            "layer_sizes": self.layer_sizes,
            "learning_rate": self.lr,
            "momentum": self.momentum,
        }
