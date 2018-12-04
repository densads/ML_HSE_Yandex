import math
import numpy as np
import pandas as pd

from timeit import default_timer as timer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class dota2:

    def path(self):
        return 'C:/Prj/L7/'

    def load_train_data(self):
        return pd.read_csv('%s%s' % (self.path(),'features.csv'), index_col='match_id',
                           encoding='utf-8')

    def load_test_data(self):
        return pd.read_csv('%s%s' % (self.path(),'features_test.csv'), index_col='match_id',
                           encoding='utf-8')

    def delete_test_columns(self, data):
        columns_exclude = ['radiant_win', 'start_time', 'tower_status_radiant', 'tower_status_dire', 'duration',
                           'barracks_status_radiant', 'barracks_status_dire']
        return data.drop(columns_exclude, axis=1)

    def delete_lgr_colums(self, data):
        columns_exclude = [
            'lobby_type',
            'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
            'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'
        ]
        return data.drop(columns_exclude, axis=1)

    def pca_transform(self, X, columns, n_comps=10):
        pca = PCA(n_components=n_comps)
        Xt=pca.fit_transform(X)
        print('PCA', pca.explained_variance_ratio_)
        print('PCA Components\n:',pca.components_[0].T)
        pca_idx = np.argmax(pca.components_[0])+1
        print('PCA max_feature_index',pca_idx,columns)
        return Xt

    #Градиентный бустинг
    def gbm(self, X, y, kf, trees, optimized=False):
        results = []
        for n_tree in trees:
            t_start = timer()
            print('START GBM', str(n_tree), 'trees')
            if not optimized:
                #Standart variant
                clf = GradientBoostingClassifier(n_estimators=n_tree, learning_rate=0.1, max_depth=3)
            else:
                # Optimized speed-up variant
                clf = GradientBoostingClassifier(n_estimators=n_tree, max_features='sqrt', learning_rate=0.1, max_depth=3)
            result = cross_val_score(clf, X, y, scoring='roc_auc', cv=kf)
            t_end = timer()
            print(str(n_tree), 'trees results mean:', result.mean(), 'raw', result)
            print('END GBM', str(n_tree), 'trees duration', round(t_end - t_start, 2), 'sec.')  # Time in seconds
            results.append(result.mean())
        print(results)

    #Градиентный бустинг с ручным расчетом auc_roc (для проверки)
    def gbm_extended(self, X, y, kf, trees, optimized=False):
        t_start = timer()
        for n_tree in trees:
            print('START GBM extended', str(n_tree), '')
            results_extended = np.array([])
            if not optimized:
                #Standart variant
                clf = GradientBoostingClassifier(n_estimators=n_tree, learning_rate=0.1, max_depth=3)
            else:
                # Optimized speed-up variant
                clf = GradientBoostingClassifier(n_estimators=n_tree, max_features='sqrt', learning_rate=0.1, max_depth=3)
            i = 1
            for train_index, test_index in kf.split(X):
                # print(train_index,test_index)
                t2_start = timer()
                print('Cross-validation', i)
                i += 1
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                # y_predict = clf.predict(X_test)
                y_predict = clf.predict_proba(X_test)[:, 1]
                results_extended = np.append(results_extended, (roc_auc_score(y_test, y_predict)))
                # print(results_extended)
                t2_end = timer()
                print('... takes', round(t2_end - t2_start, 2), 'sec.', 'raw', results_extended)
            print(str(n_tree), 'trees results mean:', results_extended.mean(), 'raw', results_extended)
            t_end = timer()
            print('END GBM extended', str(n_tree), 'trees duration', round(t_end - t_start, 2),'sec.')  # Time in seconds

    #Логистическая регрессия
    def lgr(self, data, y, kf, n_Cs, b_colums_exclude=False):
        if b_colums_exclude:
            data = self.delete_lgr_colums(data)
        print('Train data dimension:', data.shape)
        print('(LGR): Columns list to train', list(data.columns.values))
        X = data.values
        # Важно: не забывайте, что линейные алгоритмы чувствительны к масштабу признаков!
        # Может пригодиться sklearn.preprocessing.StandartScaler.
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)
        results = []
        for n_c in n_Cs:
            t_start = timer()
            print('START LGR', str(n_c), 'C')
            clf = LogisticRegression(C=n_c, solver='liblinear', penalty='l2', random_state=241)
            result = cross_val_score(clf, X_sc, y, scoring='roc_auc', cv=kf)
            t_end = timer()
            print(str(n_c), 'C results mean:', result.mean(), 'raw', result)
            print('END LGR', str(n_c), 'C duration', round(t_end - t_start, 2), 'sec.')  # Time in seconds
            results.append(result.mean())
        print(results)

    def lgr_extended(self, data, y, kf, n_Cs, b_colums_exclude=False):
        if b_colums_exclude:
            data = self.delete_lgr_colums(data)
        print('Train data dimension:', data.shape)
        print('(LGR): Columns list to train', list(data.columns.values))
        X = data.values
        # Важно: не забывайте, что линейные алгоритмы чувствительны к масштабу признаков!
        # Может пригодиться sklearn.preprocessing.StandartScaler.
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)
        results = []
        for n_c in n_Cs:
            t_start = timer()
            print('START LGR extended', str(n_c), 'C')
            results_extended = np.array([])
            clf = LogisticRegression(C=n_c, solver='liblinear', penalty='l2', random_state=241)
            i = 1
            for train_index, test_index in kf.split(X_sc):
                # print(train_index,test_index)
                t2_start = timer()
                print('Cross-validation', i)
                i += 1
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                # y_predict = clf.predict(X_test)
                y_predict = clf.predict_proba(X_test)[:, 1]
                results_extended = np.append(results_extended, (roc_auc_score(y_test, y_predict)))
                # print(results_extended)
                t2_end = timer()
                print('... takes', round(t2_end - t2_start, 2), 'sec.', 'raw', results_extended)
            print(str(n_c), 'C results mean:', results_extended.mean(), 'raw', results_extended)
            t_end = timer()
            print('END LGR extended', str(n_c), 'C duration', round(t_end - t_start, 2), 'sec.')  # Time in seconds

    #One-hot кодирование признаков героев
    def heroes_one_hot_encode(self, data, test_heroes=False, from_file=False):
        sstring='train'
        if test_heroes:
            sstring='test'
        if from_file:
            data = pd.read_csv('%s%s%s' % (self.path(),sstring ,'_heroes_encoded.csv'), index_col='match_id', encoding='utf-8')  #
        else:
            t_start = timer()
            print('START heroes encoding',sstring)
            N = 113 #num of heroes
            for i in range(1, N+1):
                r_name = 'ra' + str(i) + '_hero'
                d_name = 'da' + str(i) + '_hero'
                data[r_name] = 0
                data[d_name] = 0
            for index, row in data.iterrows():
                for p in range(1, 6):
                    r_name = 'r' + str(p) + '_hero'
                    d_name = 'd' + str(p) + '_hero'
                    r_column = 'ra' + str(data.at[index, r_name]) + '_hero'
                    d_column = 'da' + str(data.at[index, d_name]) + '_hero'
                    data.at[index, r_column] = 1
                    data.at[index, d_column] = -1
            print('Train data dimension:', data.shape)
            data.to_csv('%s%s%s' % (self.path(),sstring ,'_heroes_encoded.csv'))
            t_end = timer()
            print('END heroes encoding',sstring, round(t_end - t_start, 2), 'sec.')  # Time in seconds
        return data

    def lgr_train(self, X_sc, y, n_c=0.01):
        t_start = timer()
        print('START LGR BOW Learning')
        clf = LogisticRegression(C=n_c, solver='liblinear', penalty='l2', random_state=241)
        clf.fit(X_sc, y)
        t_end = timer()
        print('END LGR BOW Learning', round(t_end - t_start, 2), 'sec.')
        return clf

def main():

    dt2 = dota2()

    # Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше
    t_start = timer()
    print('START Train data loading')
    data=dt2.load_train_data()
    print('Train data dimension:',data.shape)
    t_end = timer()
    print('END Train data loading', round(t_end - t_start, 2), 'sec.')  # Time in seconds

    #Какие признаки имеют пропуски среди своих значений?
    data_counts = data.count()
    print('Ответ 1:\n', data_counts[data_counts<data.shape[0]],sep='') #признки, имеющие пропуски

    #2. Как называется столбец, содержащий целевую переменную?
    y_name = 'radiant_win' #столбец, содержащий целевую переменную
    print('Ответ 2:',y_name)

    #Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).
    df_y = pd.DataFrame(data=data[y_name], index=data.index)  # Train results
    data = dt2.delete_test_columns(data)
    print('Train data dimension after colums deleting:',data.shape)

    #Замените пропуски на нули с помощью функции fillna()
    data = data.fillna(0)
    X = data.values #train data
    y = np.ravel(df_y.values) #train results

    #PCA
    dt2.pca_transform(X, data.columns.values.tolist())

    # Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold), не забудьте перемешать при этом выборку (shuffle=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=241)
    kf.get_n_splits(X)

    #Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации,
    #попробуйте при этом разное количество деревьев (как минимум протестируйте следующие значения для количества деревьев:
    #10, 20, 30).

    # Trees in gradient boosting
    # trees = [1,2,4,8,10,20,30,50] #для подбора оптимального числа деревьев
    trees = [30]

    #Gradient boosting
    dt2.gbm(X, y, kf, trees) #Согласно задания
    dt2.gbm(X, y, kf, trees, optimized=True)
    #dt2.gbm_extended(X, y, kf, trees, optimized=True)

    #II.
    #Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией)
    # с помощью кросс-валидации по той же схеме, которая использовалась для градиентного бустинга.
    # Подберите при этом лучший параметр регуляризации (C).
    # Какое наилучшее качество у вас получилось?
    # Как оно соотносится с качеством градиентного бустинга?
    # Чем вы можете объяснить эту разницу?
    # Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?

    #C parameter
    #n_Cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000] #Для подбора
    n_Cs = [0.01] #оптимальный

    #Логистическая регрессия с scoring=roc_auc
    dt2.lgr(data, y, kf, n_Cs, False) #Все колонки
    dt2.lgr(data, y, kf, n_Cs, True)  #Удалены колонки r1_hero...d5_hero согласно задания

    #Логистическая регрессия с roc_auc_score вручную
    #dt2.lgr_extended(data, y, kf, n_Cs, False) #Все колонки

    #Heroes
    #Сколько различных идентификаторов героев существует в данной игре?
    df_heroes = pd.read_csv('%s%s' % (dt2.path(),'dictionaries/heroes.csv'), index_col='id', encoding='utf-8')  # Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше
    uniques_heroes = np.unique(data[['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                                      'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']].values)
    print('Уникальных героев:\n',uniques_heroes,uniques_heroes.shape)

    #4
    #Воспользуемся подходом "мешок слов" для кодирования информации о героях.
    # Пусть всего в игре имеет N различных героев.
    # Сформируем N признаков, при этом i-й будет равен нулю, если i-й герой не участвовал в матче; единице, если i-й герой играл за команду Radiant;
    # минус единице, если i-й герой играл за команду Dire.
    # Ниже вы можете найти код, который выполняет данной преобразование.
    # Добавьте полученные признаки к числовым, которые вы использовали во втором пункте данного этапа.

    data = dt2.heroes_one_hot_encode(data, False, False) #False = processing, True = from previously saved file

    #Проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра регуляризации. Какое получилось качество? Улучшилось ли оно? Чем вы можете это объяснить?
    #dt2.lgr(data, y, kf, n_Cs, True)  #Удалены колонки r1_hero...d5_hero согласно задания

    # Постройте предсказания вероятностей победы команды Radiant для тестовой выборки с помощью лучшей из изученных моделей
    # (лучшей с точки зрения AUC-ROC на кросс-валидации).
    # Убедитесь, что предсказанные вероятности адекватные — находятся на отрезке [0, 1], не совпадают между собой (т.е. что модель не получилась константной).

    data = dt2.delete_lgr_colums(data)
    print('Train data dimension:', data.shape)
    print('(LGR): Columns list to train', list(data.columns.values))
    X = data.values
    # Важно: не забывайте, что линейные алгоритмы чувствительны к масштабу признаков!
    # Может пригодиться sklearn.preprocessing.StandartScaler.
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    clf = dt2.lgr_train(X_sc, y)

    data_test = dt2.load_test_data()
    print('Test data dimension:', data_test.shape)
    data_test = data_test.fillna(0)
    data_test = dt2.heroes_one_hot_encode(data_test, True, False)
    data_test = dt2.delete_lgr_colums(data_test)
    data_test = data_test.drop(['start_time'], axis=1)
    print('Test data dimension after heroes encode:', data_test.shape)
    print('(LGR): Columns list to test', list(data_test.columns.values))
    Xt = data_test.values
    Xt_sc = sc.transform(Xt)
    y_predict = clf.predict_proba(Xt_sc)[:, 1]
    #print(y_predict)
    #Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов?
    print('Минимальное и Максимальное значение прогноза на тестовой выборке: min',np.amin(y_predict),'max',np.amax(y_predict))

    df_res = pd.DataFrame(data=y_predict, index=data_test.index.values, columns=['radiant_win'])
    df_res.index.names = ['match_id']
    df_res.to_csv('%s%s' % (dt2.path(),'test_predict.csv'))

main()