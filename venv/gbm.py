import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pylab as plt


def etransform(pred):
    return 1 / (1 + np.exp(-pred))
    #return (np.exp(-pred))

def main():
    data = pd.read_csv('C:/Prj/L5/gbm-data.csv', encoding='utf-8')
    values = data.values
    X = values[:, 1:]
    Y = np.ravel(values[:, 0])
    #print(X[:10], Y[:10])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=241)

    original_params = {'n_estimators': 250, 'random_state': 241, 'verbose': 1}
    #alearning = [1, 0.5, 0.3, 0.2, 0.1]
    alearning=[0.2]
    for learning in alearning:
        params = dict(original_params)
        params.update({'learning_rate': learning})
        print(params)
        clf = GradientBoostingClassifier(**params)
        clf.fit(X_train, y_train)

        clf_probs = clf.predict_proba(X_test)
        score = log_loss(y_test, clf_probs)
        print ('score', score)

        test_score = np.empty(len(clf.estimators_))
        train_score = np.empty(len(clf.estimators_))
        #train_score = np.empty(1)

        for i, pred in enumerate(clf.staged_decision_function(X_train)):
            #train_score[i] = clf.loss_(y_train, pred)
            #print(clf.loss_(y_train, pred))
            #val = etransform(pred)
            #print(val)
            #print(etransform(pred))
            train_score[i] = log_loss(y_train, etransform(pred))
        #y_val_pred_iter = clf.staged_decision_function(X_train)
        print('train score', train_score)
        ##print('y val', y_val_pred_iter)

        for i, pred in enumerate(clf.staged_decision_function(X_test)):
            #test_score[i] = clf.loss_(y_test, pred)
            test_score[i] = log_loss(y_test, etransform(pred))
        print('test score', test_score)

        print('minimum', test_score.min(), 'index', test_score.tolist().index(test_score.min()))

        #plt.plot(test_score)
        #plt.plot(train_score)
        #plt.legend(['test score', 'train score'])
        #plt.show()

        ntree = 36
        clf2 = RandomForestClassifier(n_estimators=ntree, random_state=241)
        clf2.fit(X_train, y_train)

        clf2_probs = clf2.predict_proba(X_test)
        score = log_loss(y_test, clf2_probs)
        print('score2', score)

main()
#print(transform(1.16458852))
#print(transform(1))