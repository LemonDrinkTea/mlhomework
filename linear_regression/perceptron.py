from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import random
def loadData(pathx, pathy):
    with open(pathx, 'r') as f:
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
    with open(pathy, 'r') as f:
        lines = f.readlines()
        t_y = []

        for i in range(0, lines.__len__(), 1):
            y =eval(lines[i].split()[0])
            t_y.append(y)
        t_y = np.array(t_y)

    return t_x, t_y


w=[0.,0.,0.]
w=np.array(w)
t_x,t_y=loadData()
permutation = np.random.permutation(t_y.shape[0])
t_x = t_x[permutation,  :]
t_y = t_y[permutation]
data=t_x[:,:2]
label=np.array(t_y)
in_0=np.where(label==1.0)
plt.scatter(data[in_0,0],data[in_0,1],marker='x',color='b')
in_1=np.where(label==0.0)
plt.scatter(data[in_1,0],data[in_1,1],marker='o',color='r')


a=0.05
b=0
for d in range(1000):
    for k,x in enumerate(t_x):
        xx = sum(w*x)
        for i in range(2):
            if xx>=b and t_y[k]==0:
                w -= a * x
            if xx<b and t_y[k]==1:
                w += a * x
right=0
for k,xx in enumerate(t_x):
    index1=sum(w*xx)
    if index1>=b and t_y[k]==1:
        right+=1
    if index1<b and t_y[k]==0:
        right+=1
x=data[:,0]
y1=(-w[2]-w[0]*x)/w[1]
plt.plot(x,y1)
plt.show()
print(right/len(t_y))