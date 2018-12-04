import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():
    scoring = 'accuracy'
    train = pd.read_csv('c:/prj/l2/perceptron-train.csv', header=None)
    test = pd.read_csv('c:/prj/l2/perceptron-test.csv', header=None)
    values = train.values
    values_t = test.values
    Y = values[:, 0]
    X = values[:, 1:3]
    Yt = values_t[:, 0]
    Xt = values_t[:, 1:3]
    model = Perceptron(random_state=241)
    model.fit(X, Y)
    Yp = model.predict(Xt)
    print(accuracy_score(Yt, Yp))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Xt = scaler.transform(Xt)
    model.fit(X, Y)
    Yp = model.predict(Xt)
    print(accuracy_score(Yt, Yp))


main()
