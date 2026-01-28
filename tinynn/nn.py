from tinynn.layer import Layer
from tinynn.loss import Loss, MSELoss
from typing import List
import numpy as np

class NN:
    def __init__(self, layers: List[Layer] = None, loss_func: Loss = None):
        """
        Initialize the neural network with an empty list of layers.

        """
        self.layers = layers if layers is not None else []
        self.loss_func: Loss | None = None

    def forward(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        if self.loss_func is None:
            raise ValueError("Loss function is not defined.")

        grad = self.loss_func.backward(y_pred, y_true)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.loss_func is None:
            raise ValueError("Loss function is not defined.")

        return self.loss_func.forward(y_pred, y_true)

    def step(self, learning_rate: float = 0.1):
        """
        Update the parameters of each layer using the computed gradients and the specified learning rate.
        """

        for layer in self.layers:
            if hasattr(layer, "step"):
                layer.step(learning_rate)


