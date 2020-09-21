from numpy import *
import numpy as np
import matplotlib.pyplot as plt
M=3
with open('train/x.txt', 'r') as f:
    lines = f.readlines()
    t_x = []
    for i in range(0, lines.__len__(), 1):
        x = lines[i].split()
        x = [float(xx) for xx in x]
        x.append(1.0)
        t_x.append(x)
    t_x = np.array(t_x)
    t_x1max = np.max(t_x[:, 0])
    t_x1min = np.min(t_x[:, 0])
    t_x2max = np.max(t_x[:, 1])
    t_x2min = np.min(t_x[:, 1])
t_x[:, 0] = (t_x[:, 0] - t_x1min) / (t_x1max - t_x1min)
t_x[:, 1] = (t_x[:, 1] - t_x2min) / (t_x2max - t_x2min)
with open('train/y.txt', 'r') as f:
    lines = f.readlines()
    t_y = []
    for i in range(0, lines.__len__(), 1):
        y = int(eval(lines[i].split()[0]))
        yy = [0.] * M
        yy[y] = 1.
        t_y.append(yy)
    t_y = np.array(t_y)
data = t_x[:, :2]

label = t_y
print(t_y)
in_0 = np.where(label == [1., 0., 0.])
print(in_0)
plt.scatter(data[in_0, 0], data[in_0, 1], marker='x', color='b')
in_1 = np.where(label == [0., 1., 0.])
plt.scatter(data[in_1, 0], data[in_1, 1], marker='o', color='r')
in_2 = np.where(label == [0., 0., 1.])
plt.scatter(data[in_2, 0], data[in_2, 1], marker='s', color='g')
plt.show()