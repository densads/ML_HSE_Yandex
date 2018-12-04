#from skimage import io
import skimage
import pylab
import numpy as np
import pandas as pd
import math

from sklearn.cluster import KMeans


def v_formula(cl, b):
    return b


def main():
    image = skimage.io.imread('C:/Prj/L6/parrots.jpg')
    print(image.shape)
    #print(image[0][0])
    #print(image)
    imgf = skimage.img_as_float(image)
    #print(imgf)
    #pylab.imshow(image)
    #pylab.show()

    i=0
    k=0
    values = np.zeros(shape=(337962,6))
    for row in imgf:
        #print(row)
        j=0
        for column in row:
            #print(column)
            r = column[0]
            g = column[1]
            b = column[2]
            values[k] = [k, i, j, r, g, b]
            j+=1
            k+=1
        i+=1
    #print(values)
    X = values[:, 3:]
    #print(X)

    km = KMeans(n_clusters=11, init='k-means++', random_state=241)
    kmeans = km.fit(X)
    print(kmeans.labels_.shape)
    print(kmeans.labels_)

    #data = pd.DataFrame(data={'v': values[:, 1], 'h': values[:, 2], 'r': values[:, 3], 'g': values[:, 4], 'b': values[:, 5], 'cl': kmeans.labels_[:, 0]},index=values[:,0])

    data = pd.DataFrame(
        data={'v': values[:, 1],  'h': values[:, 2],
              'r': values[:, 3], 'g': values[:, 4], 'b': values[:, 5],
              'cl': kmeans.labels_[:] },
        index=values[:, 0])
    data.index = data.index.map(int)
    print(data)
    clusters = data.groupby(['cl']).mean()
    #clusters = data.groupby(['cl']).median()
    print(clusters)
    cldict = clusters.to_dict('index')
    #print(data[clusters.index[0] == data['cl']])
    #data['cl_mean'] = (clusters.index[0] == data['cl'])
    data['cl_mean_r'] = 0

    #print(clusters[clusters.index == 2])
    #data['cl_mean'] = data.apply(lambda row: v_formula(row['cl'], clusters[clusters.index == row['cl']]['r']), axis=1)

    imgr = np.zeros(shape=(474, 713, 3))
    i=0
    j=0
    mse = 0
    for index, row in data.iterrows():
        #print(row)
        #r = data.at[index, 'r']
        cl = data.at[index, 'cl']
        #r = clusters[clusters.index == cl]['r']
        r = cldict[cl]['r']
        g = cldict[cl]['g']
        b = cldict[cl]['b']
        #print(cl,r)
        #r = clusters[clusters.index == data.at[index, 'cl']]['r']
        #r = row['r']
        #g = row['g']
        #b = row['b']
    #    data.at[index, 'cl_mean_r'] = r
        #data.at[index, 'r']
        imgr[i, j] = [r, g, b]
        #MSE
        #print(i,j,r,g,b)
        #print(values[index])
        ro = values[index, 3]
        go = values[index, 4]
        bo = values[index, 5]
        mse = mse + math.pow((r-ro),2) + math.pow((g-go),2) + math.pow((b-bo),2)


        j+=1
        if j>712:
            i+=1
            j=0

    #for i in range (0,474):
    #    for j in range (0,713):
    #       imgr[i, j] = []
    mse = mse / (3*474*713)
    psnr = 10 * math.log10(1/mse)
    print(mse)
    print(psnr)
    #return 0
    #print(imgr)
    #imgo = skimage.img_as_uint(imgr)
    imgo = imgr
    pylab.imshow(imgo)
    pylab.show()
    #print(data.groupby(['cl']).mean())



main()