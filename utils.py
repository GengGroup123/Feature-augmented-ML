import pandas as pd
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.decomposition import PCA
import pandas as pd
from numba.core.funcdesc import FunctionDescriptor
import random


def make_class(x,n,save=False):

    #x is the train dataframe
    #n is the threshold--------------0 is bad;1 is good
    #is the excel you want to save

    c = x.iloc[:, -1] < n
    c = pd.DataFrame(c)
    c.columns = ['class']
    c_ = []
    for i in c.loc[:, 'class']:
        i = int(i)
        c_.append(i)
    c_ = pd.DataFrame(c_)
    c_.columns = ['class']
    data1 = x.iloc[:,:-1].copy()
    data_class = pd.concat([data1, c_], axis=1)
    print('the number of good AEM is',c_.sum())
    if save==True:
        data_class.to_excel('data_class.xlsx', index=False)


def make_template(l0, weight):
    # l0 is the chain
    # weight is the probility of every chain to be concated
    ll = []
    for i in l0:
        l = list(i)

        for j in range(len(l)):

            if l[j] == '*':
                l_c = l.copy()
                chosen = random.choices([0, 1], weights=weight)

                if chosen == [1]:
                    c = random.choices(l0, weights=[random.randint(0, 10) for i in range(len(l0))])
                    l_c[j] = '' + c[0] + ''
                    # print('1:'+c[0])
                    s = ''.join(l_c)
                    ll.append(s)
                else:
                    s1 = ''.join(l)
                    ll.append(s1)
    return ll


def min_max_scaler(x):

    #x is a dataframe of AEM data in pandas

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_minmax=pd.DataFrame(scaler.fit_transform(x))
    return x_minmax

def standard_scaler(x):

    #'x is a dataframe of AEM data in pandas'

    scaler=preprocessing.StandardScaler()
    x_standard=pd.DataFrame(scaler.fit_transform(x))
    x_standard.columns = x.columns
    return x_standard


def Variance_check(x):

    # check the variance of features and sort them

    v = pd.DataFrame([i for i in [x.var().values]])
    v.columns = x.columns
    v_sort =  v.sort_values(by=0, axis=1)
    return v_sort

def Variance_selector(x,m):

    #'x is a dataframe of AEM data in pandas'
    #'m (range(0,1)) is the threshold of the variance of features in AEM data dealed with min_max_scaler or standard_scaler.'

    selector=feature_selection.VarianceThreshold(threshold=m)
    x_var=pd.DataFrame(selector.fit_transform(x))
    x_var.columns= x.iloc[:, x.var().values > m].columns
    return x_var


