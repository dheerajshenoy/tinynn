import tinynn
from tinynn.loss import MSELoss
from tinynn.nn import NN
from tinynn.layer import Conv2D
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("qtAgg")


epochs = 1000
loss_fn = MSELoss()

m = Conv2D(in_channels=3, out_channels=1, kernel_size=(3, 3))

inp = np.random.randn(3, 512, 512).astype(np.float32)


for n in range(epochs):
    out = m.forward(inp)
    loss = loss_fn.forward(out, np.zeros_like(out))


fig, ax = plt.subplots(ncols=2, nrows=1)
ax[0].imshow(inp[0], cmap="jet")
ax[1].imshow(out[0, 0], cmap="jet")
plt.show()
