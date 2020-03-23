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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

inputLayer = 4
outputLayer = 3

weight = np.zeros((4,3))
#weight+= 2 * random.random() - 1
weight += -.2
bias = np.asarray([0.0,0.0,0.0])
bias += 2 * random.random() - 1
#print(weight)
#print(bias)

learningRate = 0.05

def predict(X, y, weights):
    correct = 0
    total = 0
    for i in range(X.shape[0]):
        xOut = activation(np.dot(X[i], weights) + bias)
        #xOut = np.dot(X[i], weights) + bias
        xMax = np.argmax(xOut)
        total +=1
        if xMax == y[i]:
            correct += 1
    print("The percentage correct is: ")
    print(correct / total)

def Fit (X, y, weights, bias):
    epochCount = 1
    #print(y)
    out = np.zeros_like(X)
    errorList = []
    epochList = []
    totalError = 0
    w = weights
    while epochCount < 100:
        #xOut = activation(np.dot(X, w) + bias)
        for i in range (X.shape[0]):
            correctOut = np.zeros(outputLayer)
            xOut = activation(np.dot(X[i], w) + bias)
            for j in range(3):
                temp = np.asarray([0,0,0])
                argm = np.argmax(xOut)
                #print(argm)

                if argm != y[i] and j != y[i]:
                    temp[argm] += 1
                    error = (temp - xOut)
                    #print(error)
                    totalError += error
                    w = w.T
                    w[j] += learningRate * np.sum(error) * X[i]
                    bias[j] += learningRate * np.sum(error) * bias[j]
                    w = w.T

        """
        if np.argmax(xOut) != y[i]:
            w = w.T
            w -= learningRate * squaredError * X
            w = w.T
            bias -= learningRate * squaredError * bias
        """

        totalError /= X.shape[0]



        #updateWeights(X, y, totalError)

        if epochCount % 50 == 0 or epochCount == 1:
            print ('Epoch ', epochCount, '- Total Error: ',
                 totalError)
            errorList.append(totalError)
            epochList.append(epochCount)
        epochCount +=1
    return weights

def activation(X):
    #X *= (X > 0)
    X = 1/(1 + np.exp(-X))

    return X

predict(X_train, y_train, weight)
weight = Fit(X_train, y_train, weight, bias)
predict(X_test, y_test, weight)
#print(weight)
