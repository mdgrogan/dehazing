import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("1loss.txt", delimiter=' ')
it = []
loss = []
for i in range(len(data)):
    it.append(data[i, 0])
    loss.append(data[i, 2])

plt.plot(it, loss)
plt.title('Loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()
