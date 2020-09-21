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
            x = [float(xx) / 100 for xx in x]
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
def calculatePredicateValue(t_x,w,v,wb,vb):
    h_x=[]
    bn=[]
    for x in t_x:
        b = []
        for i in range(2):
            b.append(1. / (1 + np.exp(-np.dot(w[i], x)+wb[i])))
        bn.append(b)
        y=[]
        for i in range(3):
            y.append(1. / (1 + np.exp(-np.dot(v[i], b)+vb[i])))
        h_x.append(y)
    h_x=np.array(h_x)
    bn = np.array(bn)
    return h_x,bn
def updateGradient(h_x,bn,t_x,t_y,w,v,wb,vb):

    eh=np.array([0.,0.,0.])
    for i in range(len(h_x)):
        if t_y[i] == 0:

            yh=np.array([1.0, 0.0,0.0])
            gi=h_x[i]*(1-h_x[i])*( yh-h_x[i])
            wgz=[0.,0.,0.]
            for s in range(2):
                for k in range(3):

                    wgz[s]+=v[k][s]*gi[k]
                eh[s]=bn[i][s]*(1-bn[i][s])*wgz[s]
                w[s] += a*eh[s]*t_x[i]
                wb[s]-=a*eh[s]
            for s in range(3):
                v[s] += a*gi[s] * bn[i]
                vb[s] -= a * gi[s]
        elif t_y[i] == 1:
            yh = np.array([0.0, 1.0,0.0])
            gi = h_x[i]*(1-h_x[i])*( yh-h_x[i])
            wgz = [0., 0., 0.]

            for s in range(2):
                for k in range(3):

                    wgz[s] += v[k][s] * gi[k]
                eh[s] = bn[i][s] * (1 - bn[i][s]) * wgz[s]
                w[s] += a * eh[s] * t_x[i]
                wb[s] -= a * eh[s]
            for s in range(3):
                v[s] += a*gi[s] * bn[i]
                vb[s] -= a * gi[s]
        elif t_y[i] == 2:
            yh = np.array([0.0, 0.0, 1.0])
            gi = h_x[i] * (1 - h_x[i]) * (yh - h_x[i])
            wgz = [0., 0., 0.]

            for s in range(2):
                for k in range(3):
                    wgz[s] += v[k][s] * gi[k]
                eh[s] = bn[i][s] * (1 - bn[i][s]) * wgz[s]
                w[s] += a * eh[s] * t_x[i]
                wb[s] -= a * eh[s]
            for s in range(3):
                v[s] += a * gi[s] * bn[i]
                vb[s] -= a * gi[s]
    return w,v,wb,vb
a=0.05
w=[]
wb=[]
v=[]
vb=[]
for i in range(2):
    w.append([0.]*2)
    wb.append(0.)
for i in range(3):
    v.append([0.]*2)
    vb.append(0.)
w=np.array(w)
v=np.array(v)
t_x, t_y = loadData()
permutation = np.random.permutation(t_y.shape[0])
t_x = t_x[permutation,  :]
t_y = t_y[permutation]

for i in range(100):
    h_x,bn=calculatePredicateValue(t_x,w,v,wb,vb)

    w,v,wb,vb=updateGradient(h_x,bn,t_x,t_y,w,v,wb,vb)
h_x, bn=calculatePredicateValue(t_x,w,v,wb,vb)

right=0.0

for i in range(len(h_x)):
    if h_x[i][0]>h_x[i][1] and h_x[i][0]>h_x[i][2]and t_y[i]==0:
        right+=1
    elif h_x[i][1]>h_x[i][0] and h_x[i][1]>h_x[i][2]and t_y[i]==1:
        right+=1
    elif h_x[i][2]>h_x[i][1] and h_x[i][2]>h_x[i][0] and t_y[i]==2:
        right+=1
print(right/len(t_y))