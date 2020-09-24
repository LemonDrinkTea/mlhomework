from numpy import *
import numpy as np
import matplotlib.pyplot as plt


def loadData():
    with open('ex4x.dat', 'r') as f:
        lines = f.readlines()
        t_x = []
        for i in range(0, lines.__len__(), 1):
            x = []
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
    with open('ex4y.dat', 'r') as f:
        lines = f.readlines()
        t_y = []
        for i in range(0, lines.__len__(), 1):
            t_y.append(eval(lines[i].split()[0]))
        t_y = np.array(t_y)
    return t_x, t_y


# GD
def calculatePredicateValue(t_x, w):
    h_x = []
    for x in t_x:
        h_x.append(1. / (1 + np.exp(-np.dot(w, x))))
    return h_x


def calculateGradient(h_x, t_x, t_y):
    a = [h_x[i] - t_y[i] for i in range(len(h_x))]
    b = np.array(a).reshape(len(a), 1)
    gradients = b * t_x
    gg = [0., 0., 0.]
    for g in gradients:
        gg += g
    return gg


def updategradient(w, gradients):
    return w - 0.1 * gradients


# SGD
def SGD(w, t_x):
    for j in range(30):
        for i, xx in enumerate(t_x):
            h_xx = 1. / (1 + np.exp(-np.dot(w, xx)))
            a = h_xx - t_y[i]
            gradients = a * xx
            w = w - 0.5 * gradients
        h_x = calculatePredicateValue(t_x, w)
        iterations.append(j)
        calcost(h_x)
    return w


# Newton’s method
def calculateGradientNT(h_x, t_x, t_y):
    a = [h_x[i] - t_y[i] for i in range(len(h_x))]
    b = np.array(a).reshape(len(a), 1)
    gradients = b * t_x
    gg = [0., 0., 0.]
    for g in gradients:
        gg += g
    gg /= 80
    t_xx = mat(t_x)
    h_xx = mat(h_x)
    m = np.array(h_xx * (1 - h_xx).T)
    h = m[0, 0] * t_xx.T * t_xx
    h = h / 80
    h = np.array(h.I * mat(gg).T)
    return h.reshape(1, 3)[0]


def updateWNT(h):
    return w - h


def plotdraw():
    data = t_x[:, :2]
    label = np.array(t_y)
    in_0 = np.where(label == 1.0)
    plt.scatter(data[in_0, 0], data[in_0, 1], marker='x', color='b')
    in_1 = np.where(label == 0.0)
    plt.scatter(data[in_1, 0], data[in_1, 1], marker='o', color='r')
    x = data[:, 0]
    y1 = (-w[2] - w[0] * x) / w[1]
    plt.plot(x, y1)
    plt.show()


def currency():
    right = 0
    h_x = calculatePredicateValue(t_x, w)
    for i in range(len(h_x)):
        if h_x[i] >= 0.5 and t_y[i] == 1:
            right += 1
        elif h_x[i] < 0.5 and t_y[i] == 0:
            right += 1
    print(right / len(t_y))


def plotcost():
    plt.plot(iterations, costs, color="red")
    plt.show()
def calcost(h_x):
    cost=0.
    for i in range(len(h_x)):
        cost+=t_y[i]*log(h_x[i])+(1-t_y[i])*log(1-h_x[i])
    costs.append(cost)

t_x, t_y = loadData()
w = [0., 0., 0.]
permutation = np.random.permutation(t_y.shape[0])
t_x = t_x[permutation, :]
t_y = t_y[permutation]
costs=[]
iterations=[]
# for i in range(60):
    # GD
    # h_x = calculatePredicateValue(t_x, w)
    # gradient = calculateGradient(h_x, t_x, t_y)
    # w = updategradient(w, gradient)
    # # Newton’s method
    # # h_x = calculatePredicateValue(t_x, w)
    # # h = calculateGradientNT(h_x, t_x, t_y)
    # # w = updateWNT(h)
    #
    # # 计算cost
    # iterations.append(i)
    # calcost()
# SGD
w = SGD(w, t_x)
# 绘图
plotdraw()
plotcost()
# 计算正确率
currency()

