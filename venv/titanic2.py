import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 20)
#pd.set_option('max_rows', 5)

data = pd.read_csv('c:/prj/L1/titanic.csv', index_col='PassengerId')
#datadel=data.drop(columns=['Survived'])
datax=data[['Survived','Pclass','Sex','Age','Fare']].copy()
datax=datax.dropna()

datax['bSex']=np.where(datax['Sex']=='male', '1', '0')
#print(datax.head(1000))
#print(datax.head(1000))



datay=datax['Survived'].copy()
datax=datax[['Pclass','bSex', 'Age','Fare']].copy()


#print(data[['Pclass','Sex','Age','Fare']])

print(datax.head(1000))
clf = DecisionTreeClassifier(random_state=241)
clf.fit(datax, datay)
importances = clf.feature_importances_
print(importances)