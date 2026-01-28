from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


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
        if x.ndim == 1:
            x = x[None, :]
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if grad_out.ndim == 1:
            grad_out = grad_out[None, :]
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

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
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.y * (1 - self.y)


class Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size

        self.W = (
            rng.standard_normal((out_channels, in_channels, *kernel_size)) * 0.01
        ).astype(np.float32)
        self.b = np.zeros((out_channels,), dtype=np.float32)

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            x = x[None, ...] # Allow single sample
        self.x = x

        N, C, H, W = x.shape
        OC, IC, *kernel_size = self.W.shape

        assert C == IC, "Input channels do not match."

        out_h = H - kernel_size[0] + 1
        out_w = W - kernel_size[1] + 1
        out = np.zeros((N, OC, out_h, out_w), dtype=np.float32)

        for n in range(N):
            for oc in range(OC):
                for ic in range(IC):
                    for i in range(out_h):
                        for j in range(out_w):
                            out[n, oc, i, j] += np.sum(
                                x[n, ic, i : i + kernel_size[0], j : j + kernel_size[1]]
                                * self.W[oc, ic]
                            )
                out[n, oc] += self.b[oc]

        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self.x
        N, C, H, W = x.shape
        OC, IC, kH, kW = self.W.shape
        _, _, outH, outW = grad_out.shape

        self.dW = np.zeros_like(self.W, dtype=np.float32)
        self.db = np.sum(grad_out, axis=(0,2,3)).astype(np.float32)
        dx = np.zeros_like(x, dtype=np.float32)

        for n in range(N):
            for oc in range(OC):
                for i in range(outH):
                    for j in range(outW):
                        g = grad_out[n, oc, i, j]
                        patch = x[n, :, i:i+kH, j:j+kW]

                        self.dW[oc] += g * patch
                        dx[n, :, i:i+kH, j:j+kW] += g * self.W[oc]

        return dx
