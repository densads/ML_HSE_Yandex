import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, make_scorer

def main():
    data = pd.read_csv('C:/Prj/L5/abalone.csv', encoding='utf-8')
    data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
    #print(data)
    values = data.values
    X = values[:, :8]
    Y = np.ravel(values[:, 8:])
    #print(X)
    #print(Y)
    #return 0
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kf.get_n_splits(X)
    i = 1
    results = []
    r2s = make_scorer(r2_score)
    #for i in range(1, 51):
    #    clf = RandomForestRegressor(n_estimators=i, random_state=1)
    #    cvresults = cross_val_score(clf, X, Y, cv=kf, scoring=r2s)
    #    print(i, cvresults)
    #    val = cvresults.mean()
    #    print(round(val,2), val)
    #    results.append(cvresults.mean())
    #print(results)

    for i in range (1,51):
        clf = RandomForestRegressor(n_estimators=i, random_state=1, criterion='mae')
        clresults = np.array([])
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            clf.fit(X_train, Y_train)
            Y_predict = clf.predict(X_test)
            #print ("Yp", Y_predict)
            #print("Yt", Y_test)
            #clresults.append (r2_score(Y_test, Y_predict))
            clresults = np.append(clresults, (r2_score(Y_test, Y_predict)))
        print(i, round(clresults.mean(), 3), clresults)

main()