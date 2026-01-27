from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, lr: float) -> None:
        # default: no parameters
        pass


class Dense(Layer):
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((in_dim, out_dim)) * 0.01).astype(np.float32)
        self.b = np.zeros((out_dim,), dtype=np.float32)

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # grad_out: dL/dy, shape (N, out_dim)
        print(grad_out.shape, self.x.shape)
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        grad_in = grad_out @ self.W.T
        return grad_in

    def step(self, lr: float) -> None:
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask


class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.where(self.mask, x, self.alpha * x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad_input = np.where(self.mask, 1, self.alpha)
        return grad * grad_input


class Sigmoid(Layer):
    def __init__(self):
        self.y = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.y * (1 - self.y)
