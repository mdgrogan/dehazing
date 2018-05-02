import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'normal',
        'size' : 20}
matplotlib.rc('font', **font)

data = np.genfromtxt("1loss.txt", delimiter=' ')
train_it = []
train_loss = []
for i in range(len(data)):
    train_it.append(data[i, 0])
    train_loss.append(data[i, 1])

#data = np.genfromtxt("validation.txt", delimiter=' ')
#test_it = []
#test_loss = []
#for i in range(len(data)):
#    test_it.append(data[i, 0])
#    test_loss.append(data[i, 1])
    


plt.plot(train_it, train_loss, label="training loss", linewidth=1.0)
#plt.plot(test_it, test_loss, label="testing loss", linewidth=2.0)
plt.title('Training and Validation')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()
