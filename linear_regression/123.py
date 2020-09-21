from numpy import *
import numpy as np
import random
def loadData():
    with open('ex4x.dat', 'r') as f:
        lines = f.readlines()
        t_x = []
        for i in range(0, lines.__len__(), 1):
            x = []
            x = lines[i].split()
            x = [float(xx)/100 for xx in x]
            x.append(1.0)
            t_x.append(x)

    with open('ex4y.dat', 'r') as f:
        lines = f.readlines()
        t_y = []
        for i in range(0, lines.__len__(), 1):
            t_y.append(eval(lines[i].split()[0]))

    return t_x,t_y
#GD
def calculatePredicateValue(t_x,w):
    h_x=[]
    for x in t_x:
        h_x.append(1./(1+np.exp(-np.dot(w,x))))
    return h_x

w=[0.,0.,0.]
t_x,t_y=loadData()
for i in range(1800):
    h_x=calculatePredicateValue(t_x,w)
    a=[h_x[i]-t_y[i] for i in range(len(h_x))]
    b=np.array(a).reshape(len(a),1)
    gradients=b*t_x
    gg=[0.,0.,0.]
    for g in gradients:
        gg+=g
    gg/=80
    t_xx = mat(t_x)
    h_xx = mat(h_x)
    m=np.array(h_xx*(1-h_xx).T)
    h=m[0,0]*t_xx.T*t_xx
    h=h/80
    h=np.array(h.I*mat(gg).T)
    w=w-h.reshape(1,3)[0]
right=0
print(h_x)
for i in range(len(h_x)):
    if h_x[i]>=0.5 and t_y[i]==1:
        right+=1
    elif h_x[i]<0.5 and t_y[i]==0:
        right+=1

print(right/len(t_y))
print(w)