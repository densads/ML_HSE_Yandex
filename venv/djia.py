import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def main():
    data = pd.read_csv('C:/Prj/L4/close_prices.csv', encoding='utf-8')
    djia = pd.read_csv('C:/Prj/L4/djia_index.csv', encoding='utf-8')
    pca = PCA(n_components=10)
    values = data.values
    X = values[:, 1:]
    #print(X.shape)
    #print('X', X)
    pca.fit(X)
    #print(pca.singular_values_)
    #print(pca.components_)
    results = pca.transform(X)
    print('variance', pca.explained_variance_)
    print('variance ratio', pca.explained_variance_ratio_)
    print('singular', pca.singular_values_)
    print('mean', pca.mean_)
    #print (results.shape)
    #print (results[:, :1])
    #X_new = pca.inverse_transform(results)
    #print(X_new)
    print('components', pca.components_.T)
    print('results', results)
    print(results.shape)
    print(results[:, :1])

    dvalues=djia.values
    #print(dvalues[:, 1:])

    a = results[:, 0]
    #print(a)
    b = dvalues[:, 1]
    #print(b)


    df = pd.DataFrame()
    df['a']=a
    df['b']=b
    print(df)

    print(df['a'].astype('float64').corr(df['b'].astype('float64')))
    #print(np.corrcoef(df['a'], df['b']))

    #print('X-mean', X-pca.mean_)

    #print('T', pca.components_.T)

    #data_reduced = np.dot(X - pca.mean_, pca.components_.T)
    #data_reduced = (X- pca.mean_) @ pca.components_.T
    #print('results manual', data_reduced)
    #data_original = np.dot(data_reduced, pca.components_) + pca.mean_ # inverse_transform
    #print('restored manual', data_original)
    #print(pca.inverse_transform(results)	)


    #print(pd.DataFrame(pca.components_, columns=data.columns[1:]))

    print(pca.components_[0])
    idx = np.argmax(pca.components_[0]) + 1
    print(idx)

main()