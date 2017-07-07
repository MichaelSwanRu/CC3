# 4/07/2017 w

# default djstacking + raw data


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

#from sklearn.preprocessing import LabelEncoder

PATH = 'g:/lmv/Algoritms/_MAIL_RU/CC3'
#PATH = 'D:/algorithm/mail_ru/CC3'


train = pd.read_csv(PATH + "/train.csv", sep = ';')
test = pd.read_csv(PATH + "/test.csv", sep = ';')
y = train['cardio']


# Привожу к одинаковому виду
train['num'] = 1;
test['num'] = 0;  
test['cardio'] = 2

# Создаю общую матрицу, чтобы выровнять признаки
main = train.copy()
main = main.append(test, ignore_index=True)

main['age'] = [int(x/365) for x in main['age']]

# Преобразование данных, очистка 
# Юрий

#main.loc[main['ap_lo'] < 0, 'ap_lo'] *= -1
#main.loc[main['ap_hi'] < 0, 'ap_hi'] *= -1
#main.loc[(main['ap_hi'] < main['ap_lo']) & (main['ap_lo'] > 500) , 'ap_lo'] /= 10   
#main.loc[main['ap_hi'] < 30, 'ap_hi'] *= 10
#main.loc[main['ap_hi'] > 500, 'ap_hi'] /= 10
#main.loc[main['ap_lo'] < 15, 'ap_lo'] *= 10
#main.loc[main['ap_hi'] < main['ap_lo'], ['ap_lo', 'ap_hi']] = main.loc[main['ap_hi'] < main['ap_lo'], ['ap_hi', 'ap_lo']].values


# Преобразование данных, очистка 
# Михаил

main['ap_hi']=main['ap_hi'].apply(np.abs)
main['ap_lo']=main['ap_lo'].apply(np.abs)
main['height']=main['height'].apply(int)
main['weight']=main['weight'].apply(int)
#main['height'] = main.apply(lambda row: int(row['height']), axis=1)
#main['weight'] = main.apply(lambda row: int(row['weight']), axis=1)


main.loc[(main['ap_lo'] > 200) & (main['ap_lo'] < 2100) , 'ap_lo'] /= 10 
main.loc[(main['ap_lo'] > 2100), 'ap_lo'] /= 100 
main.loc[(main['ap_lo'] > 0) & (main['ap_lo'] <= 9) , 'ap_lo'] *= 10
main.loc[main['ap_lo'] == 0, 'ap_lo'] = 50
main.loc[(main['ap_hi'] >= 400) & (main['ap_hi'] <= 2000) , 'ap_hi'] /= 10
main.loc[main['ap_hi'] > 10000, 'ap_hi'] /= 100
main.loc[main['ap_hi'] <= 20, 'ap_hi'] *= 10



# В тесте находится примерно 20 000 комбинаций 'None' в трех столбцах ['active', 'alco', 'smoke']
# Уникальные значения в столбцах ['active', 'alco', 'smoke']  - это  0 и 1
# Поэтому предлагается попробовать заменить 'None' на 0 и на 1, т.о. добавить в test еще 10 000 строк и изменить 10 000 строк

# После построения модели получим результат на 30 000 + 10 000 строках теста, найдем задублированные 'id' и усредним по ним прогноз.
def NoneToDouble(x, db, column):
    global tmp
    z = x
    if x == 'None' : 
        z = 1
        db = pd.DataFrame(db)
        db = db.T
        db[column] = 0
        tmp = tmp.append(db, ignore_index=True)
    return z

#data4 = np.zeros( (len(work_str_new), 3), dtype = int) 
#data5 = pd.DataFrame(data=data4, columns=['level', 'ID', 'weight'])


tmp = main.loc[:0, :].copy()
column = 'active'; main[column] = main.apply(lambda row: NoneToDouble(row[column], row, column), axis=1)
main = main.append(tmp.loc[1:, :], ignore_index=True)

tmp = main.loc[:0, :].copy()    
column = 'smoke'; main[column] = main.apply(lambda row: NoneToDouble(row[column], row, column), axis=1)
main = main.append(tmp.loc[1:, :], ignore_index=True)

tmp = main.loc[:0, :].copy()
column = 'alco'; main[column] = main.apply(lambda row: NoneToDouble(row[column], row, column), axis=1)
main = main.append(tmp.loc[1:, :], ignore_index=True)    
    


main.loc[main['ap_hi'] < main['ap_lo'], ['ap_lo', 'ap_hi']] = main.loc[main['ap_hi'] < main['ap_lo'], ['ap_hi', 'ap_lo']].values    
# http://mhlife.ru/prevention/checkup/blood-pressure.html
# Согласно ссылке разделим данные по диапазонам значений артериального давления

def Dap(a, b):
    ind = 0
    if a <= 90 and b <= 60: 
        z = 1; ind = 1
    if a >= 90 and a <= 120 and b >= 60 and b <= 80:
        z = 2; ind = 1
    if a >= 120 and a <= 140 and b >= 80 and b <= 90:
        z = 3; ind = 1
    if a >= 140 and a <= 160 and b >= 90 and b <= 100: 
        z = 4; ind = 1
    if a >= 160 and a <= 180 and b >= 100 and b <= 110:
        z = 5; ind = 1
    if a >= 180 and b >= 110: 
        z = 6; ind = 1
    if ind == 0: z = 0
    return z
    
#main['D_ap'] = [Dap(main.loc[i, 'ap_hi'], main.loc[i, 'ap_lo']) for i in range(len(main)) ]
main['D_ap'] = main.apply(lambda row: Dap(row['ap_hi'], row['ap_lo']), axis=1)
# P.S. необходимо обработать нестандартные значения, которые попали в ноль (Например: a <= 90 and b > 60)


main = main.apply(pd.to_numeric, errors='ignore')

categorical=['cholesterol', 'gluc', 'smoke', 'alco', 'gender']
target_name='cardio'

alpha=10 # реально его можно подбирать по валидации, главное, чтобы не 0

# это реализация сглаженного среднего 1го уровня, но в видео говорилось, что для избавления от переобучения нам надо делать еще разбиения на фолды и уже в них все то же самое сделать
# сейчас думаю, как это поинтереснее реализовать, т.к. кода у себя я не нашел
for i in categorical:
    K=main.groupby(i).size()
    mean_y=main.groupby(i)[target_name].mean()
    global_mean_y=main[target_name].mean()
    new_heuristics=(mean_y*K+global_mean_y*alpha)/(K+alpha)
    
    new_feat_name=i+'_sa'
    main[new_feat_name]=main[i].map(dict(new_heuristics))

    
    
main = main.apply(pd.to_numeric, errors='ignore')    
    
    
    

features0 = [ x for x in main.columns if x != 'id' and x != 'cardio' and x != 'num']
from sklearn.preprocessing import MinMaxScaler
main_scaled = MinMaxScaler().fit_transform(main[features0])
main_scaled = pd.DataFrame(main_scaled, columns=features0)
main_scaled_D = main_scaled.describe().T




###################################################################################################
###################################################################################################
###################################################################################################

# plot

'''
plt.style.use('ggplot')
#%matplotlib

col1 = 'age'
col2 = 'height'
plt.figure(figsize=(10, 6))

#plt.scatter(main[col1][main['cardio'] == 0],
#            main[col2][main['cardio'] == 0],
#            alpha=0.95,
#            color='green',
#            label='Good cardio')

plt.scatter(main[col1][main['cardio'] == 1],
            main[col2][main['cardio'] == 1],
            alpha=0.15,
            color='red',
            label='Bad cardio')
plt.xlabel(col1)
plt.ylabel(col2)
plt.legend(loc='best');
'''
###################################################################################################
###################################################################################################
###################################################################################################












###################################################################################################
###################################################################################################
###################################################################################################

# devide

'''
import math

number_groups = 1+3.322*np.log10(len(main))
number_groups = int(math.ceil(number_groups))

diapazon_groups = (np.max(main['height']) - np.min(main['height']))*1.0/number_groups
diapazon_groups = int(math.ceil(diapazon_groups))


import datetime
    
def super_devide_1(train, T, number_groups, diapazon_groups):
    start_time = datetime.datetime.now() 
    M = int(round(len(train)/10.0))
    MIN = np.min(train.iloc[:,T])
    for i in range(0, len(train)):
        if train.iloc[i,T] < MIN + diapazon_groups*1: train.iloc[i,T] = 0
        for j in range(1, number_groups-1):        
            if train.iloc[i,T] >= MIN + diapazon_groups*j and train.iloc[i,T] < MIN + diapazon_groups*(j+1) : train.iloc[i,T] = j
        if train.iloc[i,T] >= MIN + diapazon_groups*(number_groups-1): train.iloc[i,T] = (number_groups-1)
        if i%M == 0:
            print "progress", i/M*10, '%', 'Time elapsed:', datetime.datetime.now() - start_time 
    print "-"
    return (train)


# 2 AGE 61
T = 9; main = super_devide_1(main, T, number_groups, diapazon_groups)



import matplotlib.pyplot as plt
plt.hist(main['height'], bins = 50)

import math

number_groups = 1+3.322*np.log10(len(main))
number_groups = int(math.ceil(number_groups))

diapazon_groups = (np.max(main['height']) - np.min(main['height']))*1.0/number_groups
diapazon_groups = int(math.ceil(diapazon_groups))

print(number_groups, diapazon_groups)

main['height']=main['height'].apply(np.abs)
main['weight']=main['weight'].apply(np.abs)


MIN = np.min(main['height'])

if main.loc[0, 'weight'] < (MIN + diapazon_groups): 
    z = 0


def super_devide_2(x):
    global MIN, diapazon_groups, number_groups
    if x < (MIN + diapazon_groups): 
        z = 0
    for j in range(1, number_groups-1):        
        if x >= MIN + diapazon_groups*j and x < MIN + diapazon_groups*(j+1):
            z = j
    if x >= MIN + diapazon_groups*(number_groups-1):
        z = (number_groups-1)
    return z



main['height_new'] = main.apply(lambda row: super_devide_2(main['height']), axis=1)
'''
###################################################################################################
###################################################################################################
###################################################################################################



















###################################################################################################
###################################################################################################
###################################################################################################

# devide 2

'''
#plt.hist(main['height'], bins = 100)    
#
#uniq_height = np.sort(main['height'].unique())
#from collections import Counter
#c = Counter(main['height'])
#table = np.zeros((len(uniq_height), 2), dtype = int)
#table[:, 0] = uniq_height
#table[:, 1] = [c[i] for i in uniq_height]
##table = pd.DataFrame(table)
#k = 12




    
#from sklearn import cluster
#from sklearn.preprocessing import StandardScaler
#
#def DBSCAN_num_clusters(db):
#    N = 0
#    try:
#        X = StandardScaler().fit_transform(db)
#        dbscan = cluster.DBSCAN(eps=.09)
#        dbscan.fit(X)
#        y_pred = dbscan.labels_.astype(np.int)
#        N = len(list(set(y_pred)))
#    except:
#        N = 0    
#    return N
#    
#    
#DBSCAN_num_clusters(main['height'])

 ###################################################################   

k = 1 + 3.322*np.log10(len(main))
k = int(k+1)

table = np.zeros((len(main), 2), dtype = int)
table[:, 0] = main['ap_hi']
table[:, 1] = main['ap_lo']
#table = pd.DataFrame(table)


X = table.copy()
X = StandardScaler().fit_transform(X)

plt.figure(figsize=(5, 5))
plt.plot(X[:, 0], X[:, 1], 'bo');

# В scipy есть замечательная функция, которая считает расстояния
# между парами точек из двух массивов, подающихся ей на вход
from scipy.spatial.distance import cdist

# Прибьём рандомность и насыпем три случайные центроиды для начала

mu, sigma = 0.6, 0.1 # mean and standard deviation
de = (np.max(X[:, 0]) - np.min(X[:, 0]))/k
x = [np.min(X[:, 0]) + de*i for i in range(k)]
y = [1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (i - mu)**2 / (2 * sigma**2) ) for i in x]

import matplotlib.pyplot as plt
plt.plot(x, y, linewidth=2, color='r')
plt.show()

centroids = np.zeros((len(x), 2), dtype = int)
centroids[:, 0] = x
centroids[:, 1] = y  
     
#s = np.random.standard_cauchy(10000)
#s = s[(s>-3) & (s<3)]  # truncate distribution so it plots well
#plt.hist(s, bins=100)
#plt.show()


cent_history = []
cent_history.append(centroids)

for i in range(k):
    # Считаем расстояния от наблюдений до центроид
    distances = cdist(X, centroids)
    # Смотрим, до какой центроиде каждой точке ближе всего
    labels = distances.argmin(axis=1)

    # Положим в каждую новую центроиду геометрический центр её точек
    centroids = centroids.copy()
#    centroids[0, :] = np.mean(X[labels == 0, :], axis=0)
#    centroids[1, :] = np.mean(X[labels == 1, :], axis=0)
#    centroids[2, :] = np.mean(X[labels == 2, :], axis=0)
    centroids[i, :] = np.mean(X[labels == i, :], axis=0)
    cent_history.append(centroids)


# А теперь нарисуем всю эту красоту
plt.figure(figsize=(8, 40))
for i in range(k):
    distances = cdist(X, cent_history[i])
    labels = distances.argmin(axis=1)

    plt.subplot(12, 2, i + 1)
#    plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'bo', label='cluster #1')
#    plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'co', label='cluster #2')
#    plt.plot(X[labels == 2, 0], X[labels == 2, 1], 'mo', label='cluster #3')
    plt.plot(X[labels == i, 0], X[labels == i, 1], 'mo', label='cluster #3')
    plt.plot(cent_history[i][:, 0], cent_history[i][:, 1], 'rx')
    plt.legend(loc=0)
    plt.title('Step {:}'.format(i + 1));




from sklearn.cluster import KMeans

inertia = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
    inertia.append(np.sqrt(kmeans.inertia_))

plt.plot(range(1, 20), inertia, marker='s');
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$');


'''

###################################################################################################
###################################################################################################
###################################################################################################






categorical_columns = [c for c in main.columns if main[c].dtype.name == 'object']
numerical_columns   = [c for c in main.columns if main[c].dtype.name != 'object']
print categorical_columns
print numerical_columns


for c in main.columns:
    print main.columns.get_loc(c), main[c].name, len(main[c].unique())

'''
# part 2 Без Нормализации

features = [ x for x in main.columns if x != 'id' and x != 'cardio' and x != 'num']        
test  = main[main['num'] == 0]
train = main[main['num'] == 1]
del test['num']; del train['num']  
test = test.reset_index(drop=True)

X_train = train[features].values
y_train = train['cardio'].values
X_test = test[features].values
'''




# part 1

features = [ x for x in main.columns if x != 'id' and x != 'cardio' and x != 'num']
from sklearn.preprocessing import MinMaxScaler
main_scaled = MinMaxScaler().fit_transform(main[features])
main_scaled = pd.DataFrame(main_scaled, columns=features)
main_scaled['num'] = main['num']
test  = main_scaled[main_scaled['num'] == 0]
train = main_scaled[main_scaled['num'] == 1]
del test['num'];  del train['num']  
test = test.reset_index(drop=True)

X_train = train[features].values
y_train = y
X_test = test[features].values








###################################################################################################
###################################################################################################
###################################################################################################
# !!!!

'''
params_est = {'n_estimators': 100,
              'subsample': 0.902,
              'learning_rate': 0.076,
              'min_samples_split': 14,
              'alpha': 0.29,
              'max_depth': 9,
              'min_samples_leaf': 5,
              'loss': 'quantile',
              'verbose': 1,
              'random_state': 1}

bst1 = GradientBoostingRegressor(**params_est)
bst1.fit(X_train, y_train)


y_pred1 = bst1.predict(X_test)
y_pred1 = y_pred1 + abs(np.min(y_pred1))
def norm (x, MX, SVA): return 1/(1+1/np.exp ( (x-MX)/SVA) )      
y_pred1 = norm(y_pred1, np.mean(y_pred1), np.sqrt( np.var(y_pred1)));


answer = test.copy()
answer['y'] = y_pred1
answer = answer[['id', 'y']]

from collections import Counter
c = Counter(answer['id'])
a = answer['id'].unique()
d = answer['id']
b = answer['y']

f = [i for i in c if c[i] > 1]
e = [np.mean(answer[answer['id'] == y]['y']) for y in f]

g = []
for i in range(len(a)):
    if a[i] in f:
        g.append(e[f.index(a[i])])
    else:
        g.append(b[i])

sub_df = pd.DataFrame(data=g, columns=['TARGET'])
sub_df.to_csv('D:/algorithm/mail_ru/CC3/submit5_1.csv', index=False, header=False)

'''

###################################################################################################
###################################################################################################
###################################################################################################









###################################################################################################
###################################################################################################
###################################################################################################

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.qda import QDA

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

#multyclass
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.linear_model import SGDClassifier
import xgboost as xgb


from sklearn.lda import LDA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

'''
# для встроенных картинок
%pylab inline
# чуть покрасивше картинки:
pd.set_option('display.mpl_style', 'default')
figsize(12, 9)

import warnings
warnings.filterwarnings("ignore")

#plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Vernada' # Ubuntu

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)

# чтобы был русский шрифт
from matplotlib import rc
 
font = {'family': 'Vernada', #Droid Sans
        'weight': 'normal'}
rc('font', **font)
'''


class DjStacking(BaseEstimator, ClassifierMixin):  
    """Стэкинг моделей scikit-learn"""
    def __init__(self, models, ens_model):
        """
        Инициализация
        models - базовые модели для стекинга
        ens_model - мета-модель
        """
        self.models = models
        self.ens_model = ens_model
        self.n = len(models)
        self.valid = None        

    def Djfit(self, X, y=None, p=0.25, cv=3, err=0.001, random_state=None):
        """
        Обучение стекинга
        p - в каком отношении делить на обучение / тест
            если p = 0 - используем всё обучение!
        cv  (при p=0) - сколько фолдов использовать
        err (при p=0) - величина случайной добавки к метапризнакам
        random_state - инициализация генератора            
        """
        if (p > 0): # делим на обучение и тест
            # разбиение на обучение моделей и метамодели
            train, valid, y_train, y_valid = train_test_split(X, y, test_size=p, random_state=random_state)            
            # заполнение матрицы для обучения метамодели
            self.valid = np.zeros((valid.shape[0], self.n))
            for t, clf in enumerate(self.models):
                clf.fit(train, y_train)
                self.valid[:, t] = clf.predict(valid)               
            # обучение метамодели
            self.ens_model.fit(self.valid, y_valid)            
        else: # используем всё обучение            
            # для регуляризации - берём случайные добавки
            self.valid = err*np.random.randn(X.shape[0], self.n)            
            for t, clf in enumerate(self.models):
                # это oob-ответы алгоритмов
                self.valid[:, t] += cross_val_predict(clf, X, y, cv=cv, n_jobs=-1, method='predict')
                # но сам алгоритм надо настроить
                clf.fit(X, y)            
            # обучение метамодели
            self.ens_model.fit(self.valid, y)          
        return self

    def Djpredict(self, X, y=None):
        """
        Работа стэкинга
        """
        # заполение матрицы для мета-классификатора
        X_meta = np.zeros((X.shape[0], self.n))        
        for t, clf in enumerate(self.models):
            X_meta[:, t] = clf.predict(X)        
        a = self.ens_model.predict(X_meta)        
        return (a)
    
def Djrun(clf, X, y, label):
    a = clf.Djpredict(X)
    print (label + ' AUC-ROC  = ' + str( roc_auc_score(y, a) ))
    return roc_auc_score(y, a)

def run(clf, X, y, label):
    a = clf.predict(X)
    print (label + ' AUC-ROC  = ' + str( roc_auc_score(y, a) ))
    return roc_auc_score(y, a)

def run_a(clf, X, y, label):
    a = clf.predict(X)
    print (label + ' AUC-ROC  = ' + str( accuracy_score(y, a) ))
    return accuracy_score(y, a)
    

def pred(clf, X):
    return clf.predict(X)

def pred_p(clf, X):
    return clf.predict_proba(X)[:, 1]

def print_roc(y, a):
    return ('AUC-ROC  = ' + str( roc_auc_score(y, a) ))
    
def print_acc(y, a):
    return ('ACC  = ' + str( accuracy_score(y, a) ))    

def Djplot_rez(rez):  
    d = len(rez)    
    xticks = [x[0] for x in rez]
    aucs = [x[1] for x in rez]    
    plt.figure(figsize=(d/3+5, 6))
    plt.bar(np.arange(len(aucs)), aucs, color='#0000AA') # , label=u'-'
    plt.xticks(np.arange(len(aucs))+0.1, xticks, rotation=90)
    plt.plot(np.arange(0, d), np.max(aucs) + 0*np.arange(0, d), c='black', label='best base')
    plt.plot(np.arange(0, d), np.min(aucs) + 0*np.arange(0, d), c='red', label='worst base')    
    plt.xlim([0, d])
    plt.ylim([np.min(aucs)-np.min(aucs)*0.01, np.max(aucs)+np.max(aucs)*0.01])
    plt.ylabel('AUC ROC')
    plt.legend(loc=3)   
    
#print_roc(test_y, pred(LR, test_X))
#print_acc(test_y, pred(LR, test_X))

random_state = 2017
verbose = 1

train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, train_size=0.7, random_state=random_state)

  


'''
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
# apply same transformation to test data
X_test = scaler.transform(X_test)  


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(train_X)  
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

'''

    


    
    
    
# Базовые алгоритмы
N = X_train.shape[1]    
rez = []
models = []
#stack = np.zeros( (len(test_y), 1), dtype = float)
#stack = pd.DataFrame(stack, columns = ['y'])
#stack['y'] = test_y

stack2 = np.zeros( (len(train_y), 1), dtype = float)
stack2 = pd.DataFrame(stack2, columns = ['y'])
stack2['y'] = train_y

stack = np.zeros( (len(test_y), 1), dtype = float)
stack = pd.DataFrame(stack, columns = ['y'])
stack['y'] = test_y

#*****************************************  Ridge ***********************************

alpha_array = np.logspace(-3, 2, num=6)
rg = Ridge(random_state=random_state)
grid = GridSearchCV(rg, param_grid={'alpha': alpha_array})
grid.fit(train_X, train_y)
best_alpha = grid.best_estimator_.alpha
rg = Ridge(alpha = best_alpha)
rg.fit(train_X, train_y)
rg_auc = run(rg, test_X, test_y, 'ridge')
rez.append(('rg', rg_auc))
stack['rg'] = pred(rg, test_X)
models.append(rg)
print 'TEST: ' + print_roc(test_y, pred(rg, test_X))
stack2['rg'] = pred(rg, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(rg, train_X))


#rg = Ridge(random_state=random_state)
#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
#alpha_array = np.logspace(-3, 2, num=6)
#param_grid={'alpha': alpha_array}
#rg = GridSearchCV(rg, param_grid=param_grid, n_jobs=-1, cv=skf, verbose=verbose, scoring='roc_auc')
#rg.fit(train_X, train_y)
#rg_auc = run(rg, test_X, test_y, 'ridge')
#rez.append(('rg', rg_auc))
#stack['rg'] = pred(rg, test_X)
#models.append(rg)


#*****************************************  LogisticRegressionCV ***********************************

#LRCV = LogisticRegressionCV(random_state=random_state, class_weight= 'balanced', n_jobs=-1)
#C_array = np.logspace(-4, 1, num=6)
#parameters = {'Cs': C_array}
#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
#LRCV = GridSearchCV(LRCV, parameters, cv=skf, verbose=verbose, n_jobs=-1)
#LRCV.fit(train_X, train_y)
##y_test = grid.predict_proba(test_X)[:, 1]
#LRCV_auc = run(LRCV, test_X, test_y, 'LRCV')
#rez.append(('LRCV', LRCV_auc))
#stack['LRCV'] = pred(LRCV, test_X)
#
#print_roc(test_y, pred_p(LRCV, test_X))


#*****************************************  LogisticRegression ***********************************

#    LogisticRegression

c_values = [x/10.0 for x in range(90,110)]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
LRCV = LogisticRegressionCV(Cs=c_values, cv=skf, verbose=verbose, n_jobs=-1)
LRCV.fit(train_X, train_y)
LRCV_auc = run(LRCV, test_X, test_y, 'LRCV')
rez.append(('LRCV', LRCV_auc))
models.append(LRCV)
stack['LRCV'] = pred(LRCV, test_X)
stack['LRCV_'] = pred_p(LRCV, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(LRCV, test_X))
stack2['LRCV'] = pred(LRCV, train_X)
stack2['LRCV_'] = pred_p(LRCV, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(LRCV, train_X))

#*****************************************  KNeighborsClassifier ***********************************



n_neighbors_array = list(np.arange(1, N))
leaf_size_array = list(np.arange(1, N))
knnC = KNeighborsClassifier()
grid = GridSearchCV(knnC, param_grid={'n_neighbors': n_neighbors_array, 'leaf_size' : leaf_size_array})
grid.fit(train_X, train_y)
best_n_neighbors = grid.best_estimator_.n_neighbors
best_leaf_size = grid.best_estimator_.leaf_size
knnC = KNeighborsClassifier(n_neighbors=best_n_neighbors, leaf_size = best_leaf_size)
knnC.fit(train_X, train_y)
knnC_auc = run(knnC, test_X, test_y, 'knnC')
rez.append(('knnC', knnC_auc))
models.append(knnC)
stack['knnC'] = pred(knnC, test_X)
stack['knnC_'] = pred_p(knnC, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(knnC, test_X))
stack2['knnC'] = pred(knnC, train_X)
stack2['knnC'+'_'] = pred_p(knnC, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(knnC, train_X))


cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
knnC2 = KNeighborsClassifier(n_neighbors=best_n_neighbors, leaf_size = best_leaf_size)
knnC2_auc = cross_val_score(knnC2, train_X, train_y, cv=cv, scoring='roc_auc').max()
knnC2.fit(train_X, train_y)
knnC2_auc = run(knnC2, test_X, test_y, 'knnC2')
rez.append(('knnC2', knnC2_auc))
stack['knnC2'] = pred(knnC2, test_X)
models.append(knnC2)
stack['knnC2_'] = pred_p(knnC2, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(knnC2, test_X))
stack2['knnC2'] = pred(knnC2, train_X)
stack2['knnC2'+'_'] = pred_p(knnC2, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(knnC2, train_X))


#*****************************************  KNeighborsRegressor ***********************************

n_neighbors_array = list(np.arange(1, N))
leaf_size_array = list(np.arange(1, N))
knnR = KNeighborsRegressor()
grid = GridSearchCV(knnR, param_grid={'n_neighbors': n_neighbors_array, 'leaf_size' : leaf_size_array})
grid.fit(train_X, train_y)
best_n_neighbors = grid.best_estimator_.n_neighbors
best_leaf_size = grid.best_estimator_.leaf_size
knnR = KNeighborsRegressor(n_neighbors=best_n_neighbors, leaf_size = best_leaf_size)
knnR.fit(train_X, train_y)
knnR_auc = run(knnR, test_X, test_y, 'knnR')
rez.append(('knnR', knnR_auc))
models.append(knnR)
print 'TEST: ' + print_roc(test_y, pred(knnR, test_X))
stack['knnR'] = pred(knnR, test_X)
stack2['knnR'] = pred(knnR, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(knnR, train_X))



cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
knnR2 = KNeighborsRegressor(n_neighbors=best_n_neighbors, leaf_size = best_leaf_size)
knn_auc = cross_val_score(knnR2, train_X, train_y, cv=cv, scoring='roc_auc').max()
knnR2.fit(train_X, train_y)
knnR2_auc = run(knnR2, test_X, test_y, 'knnR2')
rez.append(('knnR2', knnR2_auc))
stack['knnR2'] = pred(knnR2, test_X)
print 'TEST: ' + print_roc(test_y, pred(knnR2, test_X))
models.append(knnR2)
stack2['knnR2'] = pred(knnR2, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(knnR2, train_X))


#*****************************************  RandomForestRegressor ***********************************

max_depth_array = list(np.arange(2, 5))
n_estimators_array = list(np.arange(1, N))
RFR = RandomForestRegressor(n_jobs=-1, verbose=verbose, random_state=random_state)
grid = GridSearchCV(RFR, param_grid={'n_estimators': n_estimators_array, 'max_depth': max_depth_array})
grid.fit(train_X, train_y)
best_n_estimators = grid.best_estimator_.n_estimators
best_max_depth = grid.best_estimator_.max_depth
RFR = RandomForestRegressor(n_estimators = best_n_estimators, max_depth = best_max_depth)
RFR.fit(train_X, train_y)
RFR_auc = run(RFR, test_X, test_y, 'RFR')
rez.append(('RFR', RFR_auc))
stack['RFR'] = pred(RFR, test_X)
print 'TEST: ' + print_roc(test_y, pred(RFR, test_X))
models.append(RFR)
stack2['RFR'] = pred(RFR, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(RFR, train_X))

cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
RFR2 = RandomForestRegressor(n_estimators=best_n_estimators, n_jobs=-1, verbose=verbose, max_depth=best_max_depth, random_state=random_state)
RFR2_auc = cross_val_score(RFR2, train_X, train_y, cv=cv, scoring='roc_auc').max()
RFR2.fit(train_X, train_y)
RFR2_auc = run(RFR2, test_X, test_y, 'RFR2')
rez.append(('RFR2', RFR2_auc))
stack['RFR2'] = pred(RFR2, test_X)
models.append(RFR2)
print 'TEST: ' + print_roc(test_y, pred(RFR2, test_X))
stack2['RFR2'] = pred(RFR2, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(RFR2, train_X))

#*****************************************  RandomForestClassifier ***********************************

max_depth_array = list(np.arange(2, 4))
n_estimators_array = list(np.arange(1, N))
RFC = RandomForestClassifier(n_jobs=-1, verbose=verbose, random_state=random_state)
grid = GridSearchCV(RFC, param_grid={'n_estimators': n_estimators_array, 'max_depth': max_depth_array})
grid.fit(train_X, train_y)
best_n_estimators = grid.best_estimator_.n_estimators
best_max_depth = grid.best_estimator_.max_depth
RFC = RandomForestClassifier(n_estimators = best_n_estimators, max_depth = best_max_depth)
RFC.fit(train_X, train_y)
RFC_auc = run(RFC, test_X, test_y, 'RFR')
rez.append(('RFC', RFC_auc))
stack['RFC'] = pred(RFC, test_X)
models.append(RFC)
stack['RFC_'] = pred_p(RFC, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(RFC, test_X))
stack2['RFC'] = pred(RFC, train_X)
stack2['RFC'+'_'] = pred_p(RFC, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(RFC, train_X))


cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
RFC2 = RandomForestClassifier(n_estimators=best_n_estimators, n_jobs=-1, verbose=verbose, max_depth=best_max_depth, random_state=random_state)
RFC2_auc = cross_val_score(RFC2, train_X, train_y, cv=cv, scoring='roc_auc').max()
RFC2.fit(train_X, train_y)
RFC2_auc = run(RFC2, test_X, test_y, 'RFC2')
rez.append(('RFR2', RFC2_auc))
stack['RFC2'] = pred(RFC2, test_X)
models.append(RFC2)
stack['RFC2_'] = pred_p(RFC2, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(RFC2, test_X))
stack2['RFC2'] = pred(RFC2, train_X)
stack2['RFC2'+'_'] = pred_p(RFC2, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(RFC2, train_X))


#*****************************************  GradientBoostingRegressor ***********************************

n_estimators_array = list(np.arange(1, N))
gbr = GradientBoostingRegressor(verbose=verbose)
learning_rate_array = [0.001, 0.01, 0.1, 1]
grid = GridSearchCV(gbr, param_grid={'n_estimators': n_estimators_array, 'learning_rate': learning_rate_array})
grid.fit(train_X, train_y)
best_n_estimators = grid.best_estimator_.n_estimators
best_learning_rate = grid.best_estimator_.learning_rate
gbr = GradientBoostingRegressor(n_estimators = best_n_estimators, learning_rate = best_learning_rate)
gbr.fit(train_X, train_y)
gbr_auc = run(gbr, test_X, test_y, 'gbr')
rez.append(('gbr', gbr_auc))
models.append(gbr)
print 'TEST: ' + print_roc(test_y, pred(gbr, test_X))
stack['gbr'] = pred(gbr, test_X)
stack2['gbr'] = pred(gbr, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(gbr, train_X))


cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
gbr2 = GradientBoostingRegressor(n_estimators=best_n_estimators, verbose=verbose, learning_rate=best_learning_rate, random_state=random_state)
gbr2_auc = cross_val_score(gbr2, train_X, train_y, cv=cv, scoring='roc_auc').max()
gbr2.fit(train_X, train_y)
gbr2_auc = run(gbr2, test_X, test_y, 'gbr2')
rez.append(('gbr2', gbr2_auc))
stack['gbr2'] = pred(gbr2, test_X)
models.append(gbr2)
print 'TEST: ' + print_roc(test_y, pred(gbr2, test_X))
stack2['gbr2'] = pred(gbr2, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(gbr2, train_X))



#*****************************************  GradientBoostingClassifier ***********************************

n_estimators_array = list(np.arange(1, N))
gbc = GradientBoostingClassifier()
learning_rate_array = [0.01, 0.1, 1]
grid = GridSearchCV(gbc, param_grid={'n_estimators': n_estimators_array, 'learning_rate': learning_rate_array})
grid.fit(train_X, train_y)
best_n_estimators = grid.best_estimator_.n_estimators
best_learning_rate = grid.best_estimator_.learning_rate
gbc = GradientBoostingClassifier(n_estimators = best_n_estimators, learning_rate = best_learning_rate)
gbc.fit(train_X, train_y)
gbc_auc = run(gbc, test_X, test_y, 'gbc')
rez.append(('gbc', gbc_auc))
models.append(gbc)
stack['gbc'] = pred(gbc, test_X)
stack['gbc_'] = pred_p(gbc, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(gbc, test_X))
stack2['gbc'] = pred(gbc, train_X)
stack2['gbc'+'_'] = pred_p(gbc, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(gbc, train_X))


cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
gbc2 = GradientBoostingClassifier(n_estimators=best_n_estimators, verbose=verbose, learning_rate=best_learning_rate, random_state=random_state)
gbc2_auc = cross_val_score(gbc2, train_X, train_y, cv=cv, scoring='roc_auc').max()
gbc2.fit(train_X, train_y)
gbc2_auc = run(gbc2, test_X, test_y, 'gbc2')
rez.append(('gbc2', gbc2_auc))
stack['gbc2'] = pred(gbc2, test_X)
models.append(gbc2)
stack['gbc2_'] = pred_p(gbc2, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(gbc2, test_X))
stack2['gbc2'] = pred(gbc2, train_X)
stack2['gbc2'+'_'] = pred_p(gbc2, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(gbc2, train_X))





#*****************************************  xgb.XGBClassifier ***********************************

'''
gbm1 = xgb.XGBRegressor(learning_rate=0.01, max_depth=4, n_estimators=200, nthread=-1, objective='binary:logistic')    
gbm1.fit(train_X, train_y)
gbm1_auc = run(gbm1, test_X, test_y, 'gbm-d4')
rez.append(('gbm1', gbm1_auc))
stack['gbm1'] = pred(gbm1, test_X)
models.append(gbm1)


gbm2 = xgb.XGBRegressor(learning_rate=0.01, max_depth=4, n_estimators=400, nthread=-1, objective='binary:logistic')    
gbm2.fit(train_X, train_y)
gbm2_auc = run(gbm2, test_X, test_y, 'gbm2-d4-400')
rez.append(('gbm2', gbm2_auc))
stack['gbm2'] = pred(gbm2, test_X)
models.append(gbm2)
'''

gbm3 = xgb.XGBRegressor(learning_rate=0.01, max_depth=6, n_estimators=1000, nthread=-1, objective='binary:logistic')    
gbm3.fit(train_X, train_y)
gbm3_auc = run(gbm3, test_X, test_y, 'gbm3-d6-1000')
rez.append(('gbm3', gbm3_auc))
stack['gbm3'] = pred(gbm3, test_X)
models.append(gbm3)
print 'TEST: ' + print_roc(test_y, pred(gbm3, test_X))
stack['gbm3'] = pred(gbm3, test_X)
stack2['gbm3'] = pred(gbm3, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(gbm3, train_X))


gbm4 = xgb.XGBRegressor(max_depth=5, n_estimators=500, nthread=-1)  #    , silent=False
learning_rate_array = [0.01, 0.1]
objective_array = ['binary:logistic', 'reg:linear']
grid = GridSearchCV(gbm4, param_grid={'objective': objective_array, 'learning_rate': learning_rate_array})
grid.fit(train_X, train_y)
best_objective = grid.best_estimator_.objective
best_learning_rate = grid.best_estimator_.learning_rate
gbm4 = xgb.XGBRegressor(objective = best_objective, learning_rate = best_learning_rate, max_depth=5, n_estimators=500, nthread=-1)
gbm4.fit(train_X, train_y)
gbm4_auc = run(gbm4, test_X, test_y, 'gbm4')
rez.append(('gbm4', gbm4_auc))
models.append(gbm4)
print 'TEST: ' + print_roc(test_y, pred(gbm4, test_X))
stack['gbm4'] = pred(gbm4, test_X)
stack2['gbm4'] = pred(gbm4, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(gbm4, train_X))






###################################################################################################
###################################################################################################
###################################################################################################

test2  = main[main['num'] == 0]
test2 = test2.reset_index(drop=True)

X_test = test[features].values

y_pred1 = gbm3.predict(X_test)
#y_pred1 = y_pred1 + abs(np.min(y_pred1))
#def norm (x, MX, SVA): return 1/(1+1/np.exp ( (x-MX)/SVA) )      
#y_pred1 = norm(y_pred1, np.mean(y_pred1), np.sqrt( np.var(y_pred1)));

answer = test.copy()
answer['y'] = y_pred1
answer['id'] = test2['id']
answer = answer[['id', 'y']]

from collections import Counter
c = Counter(answer['id'])
a = answer['id'].unique()
d = answer['id']
b = answer['y']

f = [i for i in c if c[i] > 1]
e = [np.mean(answer[answer['id'] == j]['y']) for j in f]

g = []
for i in range(len(a)):
    if a[i] in f:
        g.append(e[f.index(a[i])])
    else:
        g.append(b[i])

sub_df = pd.DataFrame(data=g, columns=['TARGET'])
sub_df.to_csv(PATH + '/submit6_1.csv', index=False, header=False)

#main4 = main2.groupby(['d', 't'])['mpl'].median() 



e = [np.median(answer[answer['id'] == j]['y']) for j in f]         
g = []
for i in range(len(a)):
    if a[i] in f:
        g.append(e[f.index(a[i])])
    else:
        g.append(b[i])

sub_df = pd.DataFrame(data=g, columns=['TARGET'])
sub_df.to_csv(PATH + '/submit6_11.csv', index=False, header=False)
###################################################################################################
###################################################################################################
###################################################################################################


#
#all_test.ix[:, 'TARGET'] = y_pred1*0.6 + y_pred2*0.2 + y_pred0*0.2
#sub_df = pd.DataFrame(data=all_test[['TARGET']], columns=['TARGET'])
#sub_df.to_csv('g:/lmv/Algoritms/_MAIL_RU/submit3.csv', index=False, header=False)
#0,0477267
#


#params_est = {'n_estimators': 430,
#              'subsample': 0.978,
#              'learning_rate': 0.086,
#              'min_samples_split': 19.0,
#              'max_depth': 6,
#              'min_samples_leaf': 10.0,
#              'loss': 'lad',
#              'verbose': 1,
#              'random_state': 1}
























'''
gbm4 = xgb.XGBRegressor(learning_rate=0.01, max_depth=5, n_estimators=500, nthread=-1, objective='binary:logistic')    
gbm4.fit(train_X, train_y)
gbm4_auc = run(gbm4, test_X, test_y, 'gbm4-d5-500')
rez.append(('gbm4', gbm4_auc))
stack['gbm4'] = pred(gbm4, test_X)
models.append(gbm4)
'''



'''
gbm1 = xgb.XGBClassifier(learning_rate=0.05, max_depth=2, n_estimators=200, nthread=-1, objective='binary:logistic')    
gbm1.fit(train_X, train_y)
gbm1_auc = run(gbm1, test_X, test_y, 'gbm-d2')
rez.append(('gbm1', gbm1_auc))
stack['gbm1'] = pred(gbm1, test_X)
models.append(gbm1)

gbm2 = xgb.XGBClassifier(learning_rate=0.05, max_depth=5, n_estimators=200, nthread=-1, objective='binary:logistic')    
gbm2.fit(train_X, train_y)
gbm2_auc = run(gbm2, test_X, test_y, 'gbm-d5')    
rez.append(('gbm2', gbm2_auc))
stack['gbm2'] = pred(gbm2, test_X)
models.append(gbm2)
    
gbm3 = xgb.XGBClassifier(learning_rate=0.01, max_depth=2, n_estimators=200, nthread=-1, objective='binary:logistic')    
gbm3.fit(train_X, train_y)
gbm3_auc = run(gbm3, test_X, test_y, 'gbm3-d2')
rez.append(('gbm3', gbm3_auc))
stack['gbm3'] = pred(gbm3, test_X)
models.append(gbm3)

gbm4 = xgb.XGBClassifier(learning_rate=0.01, max_depth=5, n_estimators=200, nthread=-1, objective='binary:logistic')    
gbm4.fit(train_X, train_y)
gbm4_auc = run(gbm4, test_X, test_y, 'gbm4-d5')    
rez.append(('gbm4', gbm4_auc))
stack['gbm4'] = pred(gbm4, test_X)
models.append(gbm4)

gbm5 = xgb.XGBClassifier(learning_rate=0.01, max_depth=5, n_estimators=400, nthread=-1, objective='binary:logistic')    
gbm5.fit(train_X, train_y)
gbm5_auc = run(gbm5, test_X, test_y, 'gbm5-d5')    
rez.append(('gbm5', gbm5_auc))
stack['gbm5'] = pred(gbm5, test_X)
models.append(gbm5)
'''
 

#*****************************************  SVC ***********************************

C_array = np.logspace(-3, 3, num=7)
gamma_array = np.logspace(-5, 2, num=8)
svc = SVC(kernel='rbf', probability=True)
grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array})
grid.fit(train_X, train_y)
svc = SVC(kernel='rbf', C=grid.best_estimator_.C, gamma=grid.best_estimator_.gamma, probability=True)
svc.fit(train_X, train_y)
svc_auc = run(svc, test_X, test_y, 'svc1')    
rez.append(('svc', svc_auc))
stack['svc'] = pred(svc, test_X)
models.append(svc)
stack['svc_'] = pred_p(svc, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(svc, test_X))
stack2['svc'] = pred(svc, train_X)
stack2['svc'+'_'] = pred_p(svc, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(svc, train_X))



C_array = np.logspace(-3, 3, num=7)
svc1 = SVC(kernel='linear', probability=True)
grid = GridSearchCV(svc1, param_grid={'C': C_array})
grid.fit(train_X, train_y)
#print 'CV error    = ', 1 - grid.best_score_
#print 'best C      = ', grid.best_estimator_.C
svc1 = SVC(kernel='linear', C=grid.best_estimator_.C, probability=True)
svc1.fit(train_X, train_y)
svc1_auc = run(svc1, test_X, test_y, 'svc1')    
rez.append(('svc1', svc1_auc))
stack['svc1'] = pred(svc1, test_X)
stack['svc1'+'_'] = pred_p(svc1, test_X)
models.append(svc1)
print 'TEST: ' + print_roc(test_y, pred_p(svc1, test_X))
stack2['svc1'] = pred(svc1, train_X)
stack2['svc1'+'_'] = pred_p(svc1, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(svc1, train_X))



'''
#### slow
C_array = np.logspace(-5, 2, num=3)
gamma_array = np.logspace(-5, 2, num=3)
degree_array = [2, 3]
svc2 = SVC(kernel='poly')
grid = GridSearchCV(svc2, param_grid={'C': C_array, 'gamma': gamma_array, 'degree': degree_array})
grid.fit(train_X, train_y)
print 'CV error    = ', 1 - grid.best_score_
print 'best C      = ', grid.best_estimator_.C
print 'best gamma  = ', grid.best_estimator_.gamma
print 'best degree = ', grid.best_estimator_.degree
 
svc2 = SVC(kernel='poly', C=grid.best_estimator_.C, 
          gamma=grid.best_estimator_.gamma, degree=grid.best_estimator_.degree)
svc2.fit(train_X, train_y)
svc2_auc = run(svc2, test_X, test_y, 'svc2')    
rez.append(('svc2', svc2_auc))
stack['svc2'] = pred(svc2, test_X)
models.append(svc2)
#### slow
'''

#Djplot_rez(rez)


#*****************************************  MLPClassifier ***********************************


alpha_array = np.logspace(-5, 1, num=7)
MLP = MLPClassifier(random_state=random_state)
grid = GridSearchCV(MLP, param_grid={'alpha': alpha_array})
grid.fit(train_X, train_y)
best_alpha = grid.best_estimator_.alpha
MLP = MLPClassifier(alpha = best_alpha, random_state=random_state)
MLP.fit(train_X, train_y)
MLP_auc = run(MLP, test_X, test_y, 'MLP')    
rez.append(('MLP', MLP_auc))
models.append(MLP)
stack['MLP'] = pred(MLP, test_X)
stack['MLP_'] = pred_p(MLP, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(MLP, test_X))
stack2['MLP'] = pred(MLP, train_X)
stack2['MLP'+'_'] = pred_p(MLP, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(MLP, train_X))


cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
MLP2 = MLPClassifier(alpha = best_alpha, random_state=random_state)
MLP2_auc = cross_val_score(MLP2, train_X, train_y, cv=cv, scoring='roc_auc').max()
MLP2.fit(train_X, train_y)
MLP2_auc = run(MLP2, test_X, test_y, 'MLP2')
rez.append(('MLP2', MLP2_auc))
stack['MLP2'] = pred(MLP2, test_X)
models.append(MLP2)
stack['MLP2_'] = pred_p(MLP2, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(MLP2, test_X))
stack2['MLP2'] = pred(MLP2, train_X)
stack2['MLP2'+'_'] = pred_p(MLP2, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(MLP2, train_X))




#gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)
#gbt.fit(X_train[best_features_names], y_train)

#*****************************************  ABC ***********************************           

ABC = AdaBoostClassifier()
ABC.fit(train_X, train_y)
ABC_auc = run(ABC, test_X, test_y, 'ABC-a1')    
rez.append(('ABC', ABC_auc))
models.append(ABC) 
stack['ABC'] = pred(ABC, test_X)
stack['ABC_'] = pred_p(ABC, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(ABC, test_X))
stack2['ABC'] = pred(ABC, train_X)
stack2['ABC'+'_'] = pred_p(ABC, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(ABC, train_X))



#*****************************************  QDA ***********************************

QDA.fit(train_X, train_y)
QDA_auc = run(QDA, test_X, test_y, 'QDA')    
rez.append(('QDA', QDA_auc))
models.append(QDA) 
stack['QDA'] = pred(QDA, test_X)
stack['QDA_'] = pred_p(QDA, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(QDA, test_X))
stack2['QDA'] = pred(QDA, train_X)
stack2['QDA'+'_'] = pred_p(QDA, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(QDA, train_X))



#*****************************************  QDA2 ***********************************

QDA2 = QuadraticDiscriminantAnalysis()
QDA2.fit(train_X, train_y)
QDA2_auc = run(QDA2, test_X, test_y, 'QDA2')    
rez.append(('QDA2', QDA2_auc))
models.append(QDA2) 
stack['QDA2'] = pred(QDA2, test_X)
stack['QDA2_'] = pred_p(QDA2, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(QDA2, test_X))
stack2['QDA2'] = pred(QDA2, train_X)
stack2['QDA2'+'_'] = pred_p(QDA2, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(QDA2, train_X))


#*****************************************  GNB ***********************************

GNB = GaussianNB()
GNB.fit(train_X, train_y)
GNB_auc = run(GNB, test_X, test_y, 'GNB')    
rez.append(('GNB', GNB_auc))
models.append(GNB) 
stack['GNB'] = pred(GNB, test_X)
stack['GNB'+'_'] = pred_p(GNB, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(GNB, test_X))
stack2['GNB'] = pred(GNB, train_X)
stack2['GNB'+'_'] = pred_p(GNB, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(GNB, train_X))


'''
# slow
GPC = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
GPC.fit(train_X, train_y)
GPC_auc = run(GPC, test_X, test_y, 'GPC-a1')    
rez.append(('GPC', GPC_auc))
models.append(GPC)
stack['GPC'] = pred(GPC, test_X)
# slow
'''


#*****************************************  multyclass ***********************************


#*****************************************   ***********************************

Xpca = PCA(n_components=train_X.shape[1]).fit_transform(train_X)
OVRC_pca = OneVsRestClassifier(SVC(kernel='linear', probability=True))
OVRC_pca.fit(Xpca, train_y)
OVRC_pca_auc = run(OVRC_pca, test_X, test_y, 'OVRC_pca')  
rez.append(('OVRC_pca', OVRC_pca_auc))
stack['OVRC_pca'] = pred(OVRC_pca, test_X)
stack['OVRC_pca'+'_'] = pred_p(OVRC_pca, test_X)
models.append(OVRC_pca) 
print 'TEST: ' + print_roc(test_y, pred_p(OVRC_pca, test_X))
stack2['OVRC_pca'] = pred(OVRC_pca, train_X)
stack2['OVRC_pca'+'_'] = pred_p(OVRC_pca, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(OVRC_pca, train_X))


#*****************************************   ***********************************

Xcca = CCA(n_components=train_X.shape[1]).fit(train_X, train_y).transform(train_X)
OVRC_cca = OneVsRestClassifier(SVC(kernel='linear', probability=True))
OVRC_cca.fit(Xcca, train_y)
OVRC_cca_auc = run(OVRC_cca, test_X, test_y, 'OVRC_cca')  
rez.append(('OVRC_cca', OVRC_cca_auc))
stack['OVRC_cca'] = pred(OVRC_cca, test_X)
stack['OVRC_cca'+'_'] = pred_p(OVRC_cca, test_X)
models.append(OVRC_cca) 
print 'TEST: ' + print_roc(test_y, pred_p(OVRC_cca, test_X))
stack2['OVRC_cca'] = pred(OVRC_cca, train_X)
stack2['OVRC_cca'+'_'] = pred_p(OVRC_cca, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(OVRC_cca, train_X))


#*****************************************   ***********************************

Xcca = CCA(n_components=train_X.shape[1]).fit(train_X, train_y).transform(train_X)
nOVRC_cca = OneVsRestClassifier(NuSVC(kernel='linear', nu=0.01, probability=True))
nOVRC_cca.fit(Xcca, train_y)
nOVRC_cca_auc = run(nOVRC_cca, test_X, test_y, 'nOVRC_cca')  
rez.append(('nOVRC_cca', nOVRC_cca_auc))
stack['nOVRC_cca'] = pred(nOVRC_cca, test_X)
stack['nOVRC_cca'+'_'] = pred_p(nOVRC_cca, test_X)
models.append(nOVRC_cca) 
print 'TEST: ' + print_roc(test_y, pred_p(nOVRC_cca, test_X))
stack2['nOVRC_cca'] = pred(nOVRC_cca, train_X)
stack2['nOVRC_cca'+'_'] = pred_p(nOVRC_cca, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(nOVRC_cca, train_X))


#*****************************************   ***********************************

OCC2 = OutputCodeClassifier(LinearSVC(random_state=random_state), code_size=2, random_state=random_state)
OCC2.fit(train_X, train_y)
OCC2_auc = run(OCC2, test_X, test_y, 'OCC2')  
rez.append(('OCC2', OCC2_auc))
stack['OCC2'] = pred(OCC2, test_X)
models.append(OCC2) 
print 'TEST: ' + print_roc(test_y, pred(OCC2, test_X))
stack2['OCC2'] = pred(OCC2, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(OCC2, train_X))



#*****************************************   ***********************************

OvOC2 = OneVsOneClassifier(LinearSVC(random_state=random_state))
OvOC2.fit(train_X, train_y)
OvOC2_auc = run(OvOC2, test_X, test_y, 'OvOC2')  
rez.append(('OvOC2', OvOC2_auc))
stack['OvOC2'] = pred(OvOC2, test_X)
models.append(OvOC2) 
print 'TEST: ' +  print_roc(test_y, pred(OvOC2, test_X))
stack2['knnC2'] = pred(OvOC2, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(OvOC2, train_X))


#*****************************************   ***********************************

svm5 = NuSVC(cache_size=300, degree=3, gamma='auto', kernel='rbf',nu=0.01, probability=True)
svm5.fit(train_X, train_y)
svm5_auc = run(svm5, test_X, test_y, 'svm5-d5')    
rez.append(('svm5', svm5_auc))
stack['svm5'] = pred(svm5, test_X)
stack['svm5'+'_'] = pred_p(svm5, test_X)
models.append(svm5)
print 'TEST: ' +  print_roc(test_y, pred(svm5, test_X))
stack2['svm5'] = pred(svm5, train_X)
stack2['svm5'+'_'] = pred_p(svm5, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(svm5, train_X))


#*****************************************   ***********************************
#loss : str, ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, 
#or a regression loss: ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, 
#or ‘squared_epsilon_insensitive’

SGDC =  SGDClassifier(loss='log')
SGDC.fit(train_X, train_y)
SGDC_auc = run(SGDC, test_X, test_y, 'SGDC')    
rez.append(('SGDC', SGDC_auc))
stack['SGDC'] = pred(SGDC, test_X)
stack['SGDC'+'_'] = pred_p(SGDC, test_X)
models.append(SGDC)
print 'TEST: ' +  print_roc(test_y, pred(SGDC, test_X))
stack2['SGDC'] = pred(SGDC, train_X)
stack2['SGDC'+'_'] = pred_p(SGDC, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(SGDC, train_X))


#*****************************************  DecisionTreeClassifier ***********************************

max_depth_array = list(np.arange(2, 5))
criterion = ['entropy', 'gini']
DTC = DecisionTreeClassifier(random_state=random_state)
grid = GridSearchCV(DTC, param_grid={'max_depth': max_depth_array, 'criterion': criterion})
grid.fit(train_X, train_y)
best_max_depth = grid.best_estimator_.max_depth
best_criterion = grid.best_estimator_.criterion
DTC = DecisionTreeClassifier(criterion=best_criterion, random_state=random_state, max_depth = best_max_depth)
DTC.fit(train_X, train_y)
DTC_auc = run(DTC, test_X, test_y, 'DTC')
rez.append(('DTC', DTC_auc))
stack['DTC'] = pred(DTC, test_X)
stack['DTC'+'_'] = pred_p(DTC, test_X)
models.append(DTC)
print 'TEST: ' + print_roc(test_y, pred_p(DTC, test_X))
stack2['DTC'] = pred(DTC, train_X)
stack2['DTC'+'_'] = pred_p(DTC, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(DTC, train_X))


cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
DTC2 = DecisionTreeClassifier(criterion=best_criterion, max_depth=best_max_depth, random_state=random_state)
DTC2_auc = cross_val_score(DTC2, train_X, train_y, cv=cv, scoring='roc_auc').max()
DTC2.fit(train_X, train_y)
DTC2_auc = run(DTC2, test_X, test_y, 'DTC2')
rez.append(('DTC2', DTC2_auc))
stack['DTC2'] = pred(DTC2, test_X)
models.append(DTC2)
stack['DTC2'+'_'] = pred_p(DTC2, test_X)
print 'TEST: ' + print_roc(test_y, pred_p(DTC2, test_X))
stack2['DTC2'] = pred(DTC2, train_X)
stack2['DTC2'+'_'] = pred_p(DTC2, train_X)
print 'TRAIN: ' + print_roc(train_y, pred_p(DTC2, train_X))



#*****************************************  DecisionTreeRegressor ***********************************

max_depth_array = list(np.arange(2, 3))
criterion = ['mse', 'mae']
DTR = DecisionTreeRegressor(random_state=random_state)
grid = GridSearchCV(DTR, param_grid={'max_depth': max_depth_array, 'criterion': criterion})
grid.fit(train_X, train_y)
best_max_depth = grid.best_estimator_.max_depth
best_criterion = grid.best_estimator_.criterion
DTR = DecisionTreeRegressor(criterion=best_criterion, random_state=random_state, max_depth = best_max_depth)
DTR.fit(train_X, train_y)
DTR_auc = run(DTR, test_X, test_y, 'DTR')
rez.append(('DTR', DTR_auc))
stack['DTR'] = pred(DTR, test_X)
models.append(DTR)
print 'TEST: ' + print_roc(test_y, pred(DTR, test_X))
stack2['DTR'] = pred(DTR, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(DTR, train_X))


cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
DTR2 = DecisionTreeRegressor(criterion=best_criterion, max_depth=best_max_depth, random_state=random_state)
DTR2_auc = cross_val_score(DTR2, train_X, train_y, cv=cv, scoring='roc_auc').max()
DTR2.fit(train_X, train_y)
DTR2_auc = run(DTR2, test_X, test_y, 'DTR2')
rez.append(('DTR2', DTR2_auc))
stack['DTR2'] = pred(DTR2, test_X)
models.append(DTR2)
print 'TEST: ' + print_roc(test_y, pred(DTR2, test_X))
stack2['DTR2'] = pred(DTR2, train_X)
print 'TRAIN: ' + print_roc(train_y, pred(DTR2, train_X))



'''
max_depth_array = list(np.arange(2, 10))
DTR = DecisionTreeRegressor(criterion='mae', random_state=random_state)
grid = GridSearchCV(DTR, param_grid={'max_depth': max_depth_array})
grid.fit(train_X, train_y)
best_max_depth = grid.best_estimator_.max_depth
DTR = DecisionTreeRegressor(criterion='mse', random_state=random_state, max_depth = best_max_depth)
DTR.fit(train_X, train_y)
DTR_auc = run(DTR, test_X, test_y, 'DTR')
rez.append(('DTR', DTR_auc))
stack['DTR'] = pred(DTR, test_X)
models.append(DTR)
'''



# https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py    
 
Djplot_rez(rez)




from sklearn.linear_model import RidgeCV, RidgeClassifierCV, RidgeClassifier
#http://scikit-learn.org/stable/supervised_learning.html#supervised-learning



#*******************************************   best models       ****************************************  
'''
names = [x[0] for x in rez]
aucs = [x[1] for x in rez]
d = np.mean(aucs)
if d < 0.5: d = 0.5
print (len(models), len(names), len(aucs), d)
if d > 0.8: d = 0.5
#aucs1 = [x for x in aucs if x > d]
nums1 = [aucs.index(x) for x in aucs if x > d]
nums1 = list(set(nums1))
rez = [ [names[i], aucs[i]] for i in nums1] 
#n_models = [ names[i] for i in nums1]  
models = [ models[i] for i in nums1]  
#nums2 = [ i+1 for i in nums1]
#stack = stack[nums2]

Djplot_rez(rez)
''' 


names = [x[0] for x in rez]
aucs = [x[1] for x in rez]
d = np.mean(aucs)
if d < 0.5: d = 0.5
print (len(models), len(names), len(aucs), d)
if d > 0.8: d = 0.8
#aucs1 = [x for x in aucs if x > d]
nums = [aucs.index(x) for x in aucs if x > d]
nums = list(set(nums))
rez = [ [names[i], aucs[i]] for i in nums]; 
names = [x[0] for x in rez]
models = [ models[i] for i in nums];
stack = stack[names]
stack2 = stack2[names]
Djplot_rez(rez) 



# Запускаем блендинг и стекинг    
#models = [knn1, knn2, knn5, knn15, rg1, rg2, rf1, rf2, rf3, rf4, gbm1, gbm2, gbm3, gbm4]

ens_model = Ridge()
s1 = DjStacking(models, ens_model)
s1.Djfit(train_X, train_y)
s1_auc = Djrun(s1, test_X, test_y, '1-stacking')
rez.append(('1-stacking', s1_auc))
#models.append(s1)
Djplot_rez(rez)


'''
# different ens_model`s

ens_model = Ridge(0.01)
s1 = DjStacking(models, ens_model)
s1.Djfit(train_X, train_y)
s1_auc = Djrun(s1, test_X, test_y, '1-stacking')
rez.append(('1-stacking 0.01', s1_auc))
Djplot_rez(rez)

ens_model = Ridge(0.1)
s1 = DjStacking(models, ens_model)
s1.Djfit(train_X, train_y)
s1_auc = Djrun(s1, test_X, test_y, '1-stacking')
rez.append(('1-stacking 0.1', s1_auc))
Djplot_rez(rez)

ens_model = KNeighborsRegressor(n_neighbors=20)
s1 = DjStacking(models, ens_model)
s1.Djfit(train_X, train_y)
s1_auc = Djrun(s1, test_X, test_y, '1.1-stacking')
rez.append(('1.1-stacking', s1_auc))
Djplot_rez(rez)


ens_model = Ridge()
s2 = DjStacking(models, ens_model)
s2.Djfit(train_X, train_y, p=-1)
s2_auc = Djrun(s2, test_X, test_y, '2-stacking')
rez.append(('2-stacking', s2_auc))   

Djplot_rez(rez)


'''    
#Несколько блендингов подряд

ens_model = Ridge()
s0 = DjStacking(models, ens_model)
a = 0
e = []
for t in range(10):
    s0.Djfit(train_X, train_y, p=0.4)
    a += s0.Djpredict(test_X, train_y)    
    auc = roc_auc_score(test_y, a)
    print (auc)
    e.append(auc) 
    #stack['s0_'+str(t)] = s0.Djpredict(test_X, train_y)
    #print (print_roc(test_y, s0.Djpredict(test_X, train_y)))
rez.append(('10-blend+stack', np.max(e)))   
#models.append(s0)
Djplot_rez(rez)


   
#Варьируем число фолдов

ens_model = Ridge()
s2 = DjStacking(models, ens_model)
a = 0
e1 = []
for t in range(2, 5):
    s2.Djfit(train_X, train_y, p=-1, cv=t, err=0.00)
    a = s2.Djpredict(test_X, train_y)
    auc = roc_auc_score(test_y, a)
    print (auc)
    e1.append(auc)   
    #stack['s2_'+str(t)] = s2.Djpredict(test_X, train_y)
    #print ( print_roc(test_y, s2.Djpredict(test_X, train_y)) )
rez.append(('10-folds+stack', np.max(e1)))   
#models.append(s2)  
Djplot_rez(rez)



###################################################################################################
###################################################################################################
###################################################################################################







