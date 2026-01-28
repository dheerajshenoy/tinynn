import numpy as np
import tinynn
from tinynn.nn import NN
from tinynn.layer import Dense, LeakyReLU, Sigmoid

# Hyperparameters
epochs = 5000
lr = 0.1
loss_func = tinynn.loss.MSELoss()

# Input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)  # AND logic gate

# y = np.array([[0], [1], [1], [1]], dtype=np.float32) # OR logic gate
# y = np.array([[0], [1], [1], [0]], dtype=np.float32) # XOR logic gate

# Define the neural network
nn = NN()
nn.layers = [Dense(2, 10), LeakyReLU(), Dense(10, 1), Sigmoid()]
nn.loss_func = loss_func

# Training loop
for i in range(epochs):
    y_pred = nn.forward(X)
    loss = nn.loss(y_pred, y)
    nn.backward(y, y_pred)
    nn.step(learning_rate=lr)

    if i % 500 == 0:
        print("Epoch", i, "Loss:", loss.round(3))

print("Input: ", *X)
print("Pred:", *nn.forward(X).round(3))
