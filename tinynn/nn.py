from tinynn.layer import Layer
from tinynn.loss import Loss, MSELoss
from typing import List
import numpy as np


class NN:
    def __init__(self, layers: List[Layer] = None):
        """
        Initialize the neural network with an empty list of layers.

        """
        self.layers = layers if layers is not None else []
        self.loss_func: Loss = None

    def forward(self, x: np.ndarray):
        """
        Perform a forward pass through the network.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the gradient of the loss with respect to the y_pred layer and propagate it backward through the network.
        """
        # G = dL/dZ
        # dL/dW = x.T @ G
        # dL/db = G.sum(axis=0)
        # dL/dx = G @ W.T

        if self.loss_func is None:
            raise ValueError("Loss function is not defined.")

        grad = self.loss_func.backward(y_true, y_pred)

        for layer in self.layers:
            layer.backward(grad)

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the Mean Squared Error (MSE) loss between the y_pred and the y_true.
        """
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
