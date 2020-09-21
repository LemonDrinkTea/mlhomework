from numpy import *
import numpy as np
import random


def loadData():
    with open('ex4x.dat', 'r') as f:
        lines = f.readlines()
        t_x = []
        for i in range(0, lines.__len__(), 1):
            x = lines[i].split()
            x = [float(xx) / 100 for xx in x]
            x.append(1.0)
            t_x.append(x)
        print(t_x)
        t_x = np.array(t_x)

    with open('ex4y.dat', 'r') as f:
        lines = f.readlines()
        t_y = []
        for i in range(0, lines.__len__(), 1):
            t_y.append(eval(lines[i].split()[0]))
        print(t_y)
        t_y = np.array(t_y)
    return t_x, t_y


def calculatePredicateValue(t_x, w):
    h_x = []
    for x in t_x:
        xx = []
        m = 0.0
        for i in range(2):
            m += np.exp(sum(w[i] * x))
        for i in range(2):
            xx.append(np.exp(sum(w[i] * x)) / m)
        h_x.append(xx)
    h_x = np.array(h_x)
    return h_x


def calculateGradient(h_x, t_x, t_y):
    gg = np.array([[0.] * 3, [0.] * 3])
    for i in range(len(h_x)):
        if t_y[i] == 1.0:
            gg[0] += (np.array([1.0, 0.0]) - h_x[i])[0] * t_x[i]
            gg[1] += (np.array([1.0, 0.0]) - h_x[i])[1] * t_x[i]
        else:
            gg[0] += (np.array([0.0, 1.0]) - h_x[i])[0] * t_x[i]
            gg[1] += (np.array([0.0, 1.0]) - h_x[i])[1] * t_x[i]
    return gg


def updateGradient(w, gradient):
    for i in range(2):
        w[i] += 0.2 * gradient[i]
    return w


w = []
for i in range(2):
    w.append([0.] * 3)
w = np.array(w)
t_x, t_y = loadData()
for i in range(30):
    h_x = calculatePredicateValue(t_x, w)
    gradient = calculateGradient(h_x, t_x, t_y)
    w = updateGradient(w, gradient)
right = 0
w = [[10.92169506, 11.34234807, -13.09618325],
     [-10.92169506, -11.34234807, 13.09618325]]
h_x = calculatePredicateValue(t_x, w)
for i in range(len(h_x)):
    if h_x[i][0] >= 0.5 and t_y[i] == 1:
        right += 1
    elif h_x[i][1] >= 0.5 and t_y[i] == 0:
        right += 1
print(right / len(t_y))
print(w)
