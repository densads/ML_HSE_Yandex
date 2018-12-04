import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing


def main():
    wine = pd.read_csv('c:/prj/l2/wine.data', header=None)
    values = wine.values
    Y = values[:, 0]
    X = values[:, 1:14]
    X = preprocessing.scale(X)
    #print(Y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf.get_n_splits(X)
    #model.fit(X, Y)
    scoring = 'accuracy'
    results = np.array([])
    names = []
    for n in range(1, 51):
        model = KNeighborsClassifier(n_neighbors=n)
        cvresults = cross_val_score(model, X, Y, scoring=scoring, cv=kf)
        result = cvresults.mean()
        results = np.append(results, result)
        names.append(n)
        msg = "%s: %f" % (n, result)
        print(msg)
        #print(cvresults)
    print(results)
    print(names[results.argmax()])
main()


