import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

def main():
    enc = DictVectorizer()
    vectorizer = TfidfVectorizer(input='content', lowercase=True, min_df=5)
    df_salary = pd.read_csv('C:/Prj/L4/salary-train.csv', encoding='utf-8')
    df_test = pd.read_csv('C:/Prj/L4/salary-test-mini.csv', encoding='utf-8')
    df_salary['FullDescription'] = df_salary['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    df_test['FullDescription'] = df_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    df_salary['LocationNormalized'].fillna('nan', inplace=True)
    df_salary['ContractTime'].fillna('nan', inplace=True)
    #print(df_salary.head(10))
    X_train_categ = enc.fit_transform(df_salary[['LocationNormalized', 'ContractTime']].to_dict('records'))
    X_test_categ = enc.transform(df_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
    #print(enc.vocabulary_)
    #print(enc.get_feature_names())
    #print(X_train_categ)
    #print(X_test_categ)
    #df_salary['locnum']=df_salary['LocationNormalized']
    #print(df_salary.head(10))
    X_train = vectorizer.fit_transform(df_salary['FullDescription'])
    X_test = vectorizer.transform(df_test['FullDescription'])
    X = hstack([X_train, X_train_categ])
    Xt = hstack([X_test, X_test_categ])
    clf = Ridge(alpha=1.0, random_state=241)
    y_train = df_salary['SalaryNormalized'].values
    print(y_train)
    clf.fit(X, y_train)
    ts = clf.predict(Xt)
    print (ts)
    #print (X)
    #print(vectorizer.get_feature_names())
    #tfile = open('C:/Prj/L4/salary-train.csv', 'r', encoding='utf-8')
    #tfile = 'C:/Prj/L4/salary-train.csv'
    #text = [tfile.read()]
    #tfile.close()
    #print(text)

    #print(data)



main()