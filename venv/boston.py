import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def main():
    boston = load_boston()
    X = boston["data"]
    Y = boston["target"]
    names = boston["feature_names"]
    X = preprocessing.scale(X)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf.get_n_splits(X)
    scoring = 'neg_mean_squared_error'
    results = np.array([])
    names = []
    ps=np.linspace(1.0, 10.0, num=200)
    n=0
    for n in range(1, len(ps)):
        model = KNeighborsRegressor(n_neighbors=5, weights='distance', p=n, metric='minkowski')
        cvresults = cross_val_score(model, X, Y, scoring=scoring, cv=kf)
        #print(cvresults)
        result = cvresults.mean()
        results = np.append(results, result)
        names.append(n)
        msg = "%s: %f" % (n, result)
        print(msg)
    print(results)
    print(names[results.argmax()])

main()