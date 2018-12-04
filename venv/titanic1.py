import pandas as pd
data = pd.read_csv('c:/prj/L1/titanic.csv', index_col='PassengerId')

#print(data.head())

total=len(data.index)
#print(total)

#1
#print(data.groupby(['Sex']).count())

#2
#x=len(data[data['Survived']==1].index)
#print(x/total)

#3
#x=len(data[data['Pclass']==1].index)
#print(x/total)

#4
#av=data['Age'].mean()
#m=data['Age'].median()
#print(av, m)

#5
#c=data['SibSp'].corr(data['Parch'])
#print(c)
dfem=(data[data['Sex']=='female'])

print(dfem['Name'])
dfr=pd.DataFrame(columns=['Name'])

words=['Mrs','Miss']
for ind in dfem.index:
    s=dfem['Name'][ind]
    for word in words:
        sp=s.partition(word)[2]
        sp=sp.partition('. ')[2]
        if len(sp)>0:
            sp=sp.partition(' ')[0]
            sp=sp.replace('(','').replace(')','')
            #print(sp, len(sp))
            sr=pd.Series({'Name' : sp})
            #sr = pd.Series([sp],[ind])
            #sr = pd.Series([sp],[ind])
            dfr=dfr.append(sr, ignore_index=True)
print(dfr.head(1000))

grouped=dfr.groupby(['Name'])
print(grouped['Name'].count())

#print(dfem[dfem['Name'].str.contains('Mrs')]['Name'])