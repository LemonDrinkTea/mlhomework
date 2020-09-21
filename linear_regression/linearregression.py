from numpy import *
import numpy as np
import matplotlib.pyplot as plt
def init(X,Y):
    X = np.array(X).astype(float)
    Y = np.array(Y)
    W = [0.0,0.0]
    return X,Y,W

def calculatePredicatedValue(X,W):
    h_x=[]
    xi=[]
    for x in X:
         xx=[]
         xx.append(x)
         xx.append(1.0)
         xi.append(xx)
         h_x.append(np.dot(W,xx))

    return h_x,xi

def calculateGradient(h_x,Y,xi):
    a=h_x-Y
    b=np.array(a).reshape(len(a),1)
    gradient=b*xi
    gg=[0.,0.]
    for g in gradient:
        gg+=g
    return gg

def updateW(W,gradient):
    return W-0.002*gradient

X= [0, 1, 2, 3, 4, 5, 6, 7, 8,
9, 10, 11, 12, 13]

Y=  [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704,
6.853, 7.971, 8.561, 10.000, 11.280, 12.900]
X,Y,W=init(X,Y)
# 画图
# 1.真实的点
plt.scatter(X, Y, color='blue')
h_x,xi=calculatePredicatedValue(X,W)
xi=mat(xi)
y=mat(Y)
w=(xi.T*xi).I*xi.T*y.T
w=np.array(list(w)).reshape(1,len(list(w)))
h_x,xi=calculatePredicatedValue(X,list(w[0]))
# 2.拟合的直线
plt.plot(X, h_x, color='red', linewidth=4)

X,Y,W=init(X,Y)
for i in range(300):
    h_x,xi=calculatePredicatedValue(X,W)
    gradient=calculateGradient(h_x,Y,xi)
    W=updateW(W,gradient)
print(sum(W*[14,1]))
plt.plot(X, h_x, color='red', linewidth=4)