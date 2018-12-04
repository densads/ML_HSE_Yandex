import numpy as np
import pandas as pd
from sklearn.svm import SVC


def main():
    scoring = 'accuracy'
    data = pd.read_csv('c:/prj/l3/svm-data.csv', header=None)
    values = data.values
    Y = values[:, 0]
    X = values[:, 1:3]
    #print(X)
    model = SVC(C=100000, random_state=241,  kernel='linear')
    model.fit(X, Y)
    print(model.support_vectors_)
main()