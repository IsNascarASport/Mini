import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import seaborn
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target

seed = np.random.randint(0, 500)
np.random.seed(seed)
np.random.shuffle(X)
np.random.seed(seed)
np.random.shuffle(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

inputLayer = 4
outputLayer = 3

weight = np.zeros((4,3))
weight *= 1.0
#weight+= 2 * random.random() - 1
#weight += -.2

learningRate = 0.15

reg = .005


def loss(X, y, w):
    loss = 0
    dW = np.zeros_like(weight)
    for i in range(X.shape[0]):
        scores = X[i].dot(w)
        correct = scores[y[i]]
        for j in range(3):
            if j == y[i]:
                continue
            li =  correct - w.T[j].dot(X[i]) + 1.0
            #print (li)
            margin = max(0,li)
            loss += margin
            if margin > 0:
                dW[:, y[i]] += X[i, :]
                dW[:, j] -= X[i, :]
    dW / X.shape[0]
    dW += reg * w
    loss /= X.shape[0]
    loss += 0.5 * reg * np.sum(w * w)
    return loss , dW

def sgd(dW, w):
    temp = w
    temp += -learningRate * dW * reg
    return temp

def predict(X, y, w):
    correct = 0
    total = 0
    for i in range(X.shape[0]):
        score = X[i].dot(w)
        if np.argmin(score) == y[i]:
            correct += 1
            total += 1
        else:
            total += 1
    acc = correct / total
    print("The Percent Correct is ", acc)

XList = np.array_split(X, 10)
yList = np.array_split(y, 10)
for i in range(len(XList)):
    lossi, dW = loss(XList[i], yList[i], weight)
    weight = sgd(dW, weight)
    print ("Loss at ", i + 1, " Iterations = ", lossi)
predict(X_test, y_test, weight)
