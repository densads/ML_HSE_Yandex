import os
import pandas as pd
import numpy as np
import math
import sklearn.metrics
from sklearn.metrics import roc_auc_score


def sigmoid(x):
    return 1.0 / (1 + math.exp(x))


def distance(a, b):
    return np.sqrt(np.square(a[0]-b[0])+np.square(a[1]-b[1]))


def log_reg (X, y, k, w, C, eps, max_iter):
    w1, w2 = w
    #print(w1)
    #w1new = w1 + k * np.mean(Y * X[:, 0] * (1 - (1. / 1 + np.exp(-Y * (w1 * X[:, 0] + w1 * X[:, 1]))))) - k * C * w1
    #print(w1new)
    for n in range (max_iter):
        #print(Y*X[:,0])
        #print(w1 * X[:, 0])
        #w1new = w1 + k * np.mean(y * X[:, 0] * (1 - (1. / 1 + np.exp(-y * (w1 * X[:, 0] + w2 * X[:, 1]))))) - k * C * w1
        #w2new = w2 + k * np.mean(y * X[:, 1] * (1 - (1. / 1 + np.exp(-y * (w1 * X[:, 0] + w2 * X[:, 1]))))) - k * C * w2
        s1 = 0
        s2 = 0
        l = len(X)+1
        for i in range(0,l-1):
            #s=X[i, 0]
            #print('x1=', X[i,0])
            s1 = s1 + X[i, 0] * y [i] * (
                            1 - ( 1 / (
                                1 + math.exp(
                                    - y [i] * ( w1 * X[i,0] + w2 * X[i,1]  )
                                )
                            ))
            )
            s2 = s2 + X[i, 1] * y [i] * (
                            1 - ( 1 / (
                                1 + math.exp(
                                    - y [i] * ( w1 * X[i,0] + w2 * X[i,1]  )
                                )
                            ))
            )
            #break
        w1new = w1 + (k / l) * s1 - k * C * w1
        w2new = w2 + (k / l) * s2 - k * C * w2
        print('n=',n,'w1=',w1,'w2=',w2,'w1new=',w1new,'w2new=',w2new, 'distance=', distance((w1new, w2new), (w1, w2)))

        if distance((w1new, w2new), (w1, w2)) < eps:
            break
        w1, w2 = w1new, w2new

    predictions=[]
    for i in range(len(X)):
        t1 = -w1 * X[i, 0] - w2 * X[i, 1]
        s = sigmoid(t1)
        predictions.append(s)
    return(predictions)


def main():
    raw_panda_data = pd.read_csv('c:/prj/l3/data-logistic.csv', header=None)
    #X_ = np.matrix(raw_panda_data.values[:,1:])
    #Y_ = np.matrix(raw_panda_data.values[:,:1].T[0])
    #X_ = raw_panda_data[[1,2]].as_matrix()
    #Y_ = raw_panda_data[0].as_matrix()
    num_columns = raw_panda_data.shape[1]                       # (num_rows, num_columns)
    X = raw_panda_data.iloc[:, 1:num_columns]  # [ slice_of_rows, slice_of_columns ]
    y = raw_panda_data.iloc[:, 0]
    X_ = (X.values)   # pandas.DataFrame -> numpy.ndarray -> numpy.matrix
    y_ = (y.values)   # pandas.DataFrame -> numpy.ndarray -> numpy.matrix

    print(X_)
    print(y_)

    p0 = log_reg(X_, y_, 0.1, [0.0, 0.0], 0, 0.00001, 500)
    print(p0)
    print('0', roc_auc_score(y_, p0))

    p1 = log_reg(X_, y_, 0.1, [0.0, 0.0], 10, 0.00001, 500)
    print(p1)
    print('10', roc_auc_score(y_, p1))


main()