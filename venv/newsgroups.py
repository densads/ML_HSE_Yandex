import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def main():
    scoring = 'accuracy'
    newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
    X = newsgroups.data
    Y = newsgroups.target
    #print(X[0])
    vectorizer = TfidfVectorizer()
    #vectorizer = CountVectorizer()
    analyze = vectorizer.build_analyzer()
    #print(analyze(X[0]))
    X = vectorizer.fit_transform(X)
    #print(vectorizer.vocabulary_)
    #print(vectorizer.vocabulary_.get('from'))
    feature_names = vectorizer.get_feature_names()
    #print(feature_names)
    kf = KFold(n_splits=5, shuffle=True, random_state=241)
    kf.get_n_splits(X)
    model = SVC(C=1, kernel='linear', random_state=241)
    if False:
        #cvresults = cross_val_score(model, X, Y, scoring=scoring, cv=kf)
        #print(cvresults)
        #model.fit(X, Y)
        grid = {'C': np.power(10.0, np.arange(-5, 6))}
        #grid = {'C' : 10000}
        gs = GridSearchCV(model, grid, scoring='accuracy', cv=kf)
        gs.fit(X, Y)
        print(gs.cv_results_)
        print(gs.best_score_)
        print(gs.best_params_)
        print(gs.best_estimator_)

    model.fit(X, Y)
    coef = model.coef_
    print(coef)
    coefa = coef.toarray()
    #coefs = np.argsort(coefa)
    #print(coefa[0])
    #coefb = sorted(coefa[0], key = abs)
    #print(coefb)
    coefz=zip(coefa[0], feature_names)
    print( sorted(list(zip(*sorted(coefz, key=lambda x: abs(x[0]))[-10:]))[1]) )

    #topn = coefz[-10:]

    #print(topn)

    #print(feature_names[11098])
    #print(coefs[3])
    #top10 = np.argsort(model.coef_[i])[-10:]
    #top10 = np.argsort(model.coef_)
    #print(model.classes_)
    #print(model.coef_.shape)
    #top10 = model.coef_
    #print(top10)
        #print("%s: %s" % (label, " ".join(feature_names[top10])))
    #print(model.coef_)

main()