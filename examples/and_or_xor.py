import numpy as np
import tinynn
from tinynn.nn import NN
from tinynn.layer import Dense, LeakyReLU, Sigmoid

nn = NN()
nn.layers = [Dense(2, 5), LeakyReLU(), Dense(5, 1), Sigmoid()]
nn.loss_func = tinynn.loss.MSELoss()

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [1]], dtype=np.float32)

epochs = 5000

for i in range(epochs):
    y_pred = nn.forward(X)
    loss = nn.loss(y_pred, y)
    nn.backward(y, y_pred)
    nn.step(learning_rate=0.1)

    if i % 500 == 0:
        print("Epoch", i, "Loss:", loss.round(3))

print("Input: ", *X)
print("Pred:", *nn.forward(X).round(3))
