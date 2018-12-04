import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def main():
    data = pd.read_csv('c:/prj/l3/classification.csv')
    print('tp',data[(data['true']==1) & (data['pred']==1)].shape)
    print('fp',data[(data['true'] == 0) & (data['pred'] == 1)].shape)
    print('fn', data[(data['true'] == 1) & (data['pred'] == 0)].shape)
    print('tn', data[(data['true'] == 0) & (data['pred'] == 0)].shape)
    print('accuracy_score',accuracy_score(data['true'], data['pred']))
    print('precision_score', precision_score(data['true'], data['pred']))
    print('recall_score', recall_score(data['true'], data['pred']))
    print('f1_score', f1_score(data['true'], data['pred']))

    metrics = ['score_logreg', 'score_svm', 'score_knn', 'score_tree']
    scores = pd.read_csv('c:/prj/l3/scores.csv')

    for metric in metrics:
        Y = scores[['true',metric]]
        print('roc_auc_score',metric,roc_auc_score(scores['true'],scores[metric]))

    #for metric in metrics:
    metric='score_logreg'
    for metric in metrics:
        Y = scores[['true', metric]]
        precision, recall, thresholds = precision_recall_curve(scores['true'],scores[metric])
        #print(zip(precision, recall))
        plt.plot(recall, precision, '.', label=metric)

        temp_list = []
        for n, m in zip(precision, recall):
            if m>0.7:
                temp_list.append(n)
        print(metric, max(temp_list))
    plt.legend()
    plt.show()


main()