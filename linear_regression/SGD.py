from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import random
def loadData():
    with open('train/x.txt', 'r') as f:
        lines = f.readlines()
        t_x = []
        for i in range(0, lines.__len__(), 1):
            x = lines[i].split()
            x = [float(xx) for xx in x]
            x.append(1.0)
            t_x.append(x)
        print(t_x)
        t_x = np.array(t_x)

    with open('train/y.txt', 'r') as f:
        lines = f.readlines()
        t_y = []
        for i in range(0, lines.__len__(), 1):
            t_y.append(eval(lines[i].split()[0]))
        print(t_y)
        t_y = np.array(t_y)
    return t_x,t_y


w=[]
for i in range(3):
    w.append([0.]*3)
w=np.array(w)

t_x,t_y=loadData()
permutation = np.random.permutation(t_y.shape[0])
t_x = t_x[permutation,  :]
t_y = t_y[permutation]
data=t_x[:,:2]
label=np.array(t_y)
print(t_y)
in_0=np.where(label==0)
plt.scatter(data[in_0,0],data[in_0,1],marker='x',color='b')
in_1=np.where(label==1)
plt.scatter(data[in_1,0],data[in_1,1],marker='o',color='r')
in_2=np.where(label==2)
plt.scatter(data[in_2,0],data[in_2,1],marker='s',color='r')
for j in range(1000):

    for k,x in enumerate(t_x):
        xx=[]
        m = 0.0
        for i in range(3):
            m += np.exp(sum(w[i]*x))
        for i in range(3):
            xx.append(np.exp(sum(w[i]*x))/m)
        xx=np.array(xx)

        gg = np.array([[0.] * 3, [0.] * 3,[0.] * 3])
        if t_y[k] == 0:
            gg[0] += (np.array([1, 0,0]) - xx)[0] * x
            gg[1] += (np.array([1, 0,0]) - xx)[1] * x
            gg[2] += (np.array([1, 0, 0]) - xx)[2] * x
        if t_y[k] == 1:
            gg[0] += (np.array([0, 1,0]) - xx)[0] * x
            gg[1] += (np.array([0, 1,0]) - xx)[1] * x
            gg[2] += (np.array([0, 1, 0]) - xx)[2] * x
        if t_y[k] == 2:
            gg[0] += (np.array([0, 0, 1]) - xx)[0] * x
            gg[1] += (np.array([0, 0, 1]) - xx)[1] * x
            gg[2] += (np.array([0, 0, 1]) - xx)[2] * x
        for i in range(3):
            w[i] +=0.02*gg[i]

print(w)
x=data[:,0]

y1=(-w[0][2]-w[0][0]*x)/w[0][1]
plt.plot(x,y1)
y3=(-w[2][2]-w[2][0]*x)/w[2][1]
plt.plot(x,y3)
plt.show()
right=0
for k,xx in enumerate(t_x):
    index1=[]
    index1.append(sum(w[0]*xx))
    index1.append(sum(w[1]*xx))
    index1.append(sum(w[2] * xx))
    ck = index1.index(max(index1))
    if ck == t_y[k]:
        right+=1
print(right/len(t_y))