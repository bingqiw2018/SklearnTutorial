#coding=utf-8 
'''
Created on 2018年4月6日

@author: bingqiw
'''
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from matplotlib.figure import SubplotParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.functions.LinearFunc import polynomial_model
from src.functions.LinearFunc import func_to_dist,scatter_ctgx,func_to_ctgx,func_to_logisc,scatter_fire
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

def poly_train(X_train, X_test, Y_train, Y_test):     
    model = polynomial_model(degree=2)
    model.fit(X_train,Y_train)
    train_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)
    mse = mean_squared_error(Y_test, model.predict(X_test))
    print("train_score:{:.6};test_score:{:.6};mse:{}".format(train_score,test_score,mse))

def logistic_train(X_train, Y_train,X_test, Y_test):
    cls = LogisticRegression()
    cls.fit(X_train, Y_train)
    train_score = cls.score(X_train, Y_train)
    test_score = cls.score(X_test, Y_test)
    print ("train_score:{};test_score:{}".format(train_score,test_score))
    
def decisionTree_train(X_train, Y_train,X_test, Y_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train,Y_train)
    train_score = clf.score(X_train, Y_train)
    test_score = clf.score(X_test, Y_test)
    
    print ("train_score:{};test_score:{}".format(train_score,test_score))
    joblib.dump(clf, 'd:/tmp/decisionTree_train_alien.pkl',compress=3) 
        
def loadData():
    name = 'D:/Program Files/eclipse4.6/workspace/SkLearnDemo/data/alien_data_500.xlsx'
    data_src = pd.read_excel(name,header=None,encoding='utf-8')
    print data_src.shape
#     data_src[13] = data_src.apply(lambda x: func_to_ctgx(x),axis=1)
#     data_src[14] = data_src.apply(lambda x: func_to_logisc(x),axis=1)
#     data_src[15] = data_src.apply(lambda x: func_to_dist(x,4,11),axis=1)
#     data_src[16] = data_src.apply(lambda x: func_to_dist(x,5,12),axis=1)
    
#     data_src.to_excel(name,header=None,index=False)
#     data_src = pd.read_excel(name,header=None,encoding='utf-8')
#     print data_src.shape
#     data = data_src.loc[:,[13]]
#     target = data_src.loc[:,[14]]
#     X_train, X_test, y_train, y_test = train_test_split(data, target,train_size=0.8, test_size=0.20, random_state=0)
#     cls = LogisticRegression()
#     cls.fit(X_train, y_train)
#     print('Score　Train: %.2f' % cls.score(X_train, y_train))
#     print('Score Test: %.2f' % cls.score(X_test, y_test))
    
    return data_src
    
data_src = loadData()
df_01 = data_src.loc[0:299]
df_02 = data_src.loc[300:500]
X = df_01.loc[:,[4,5,11,12,13]]
Y = df_01.loc[:,[14]]
PX = df_02.loc[:,[4,5,11,12,13]]
PY = df_02.loc[:,[14]]
print("导入数据X{},Y{}".format(X.shape,Y.shape))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,train_size=0.8, test_size=0.20, random_state=0)
print("数据划分结果：X_train{},Y_train{},X_test{},Y_test{}".format(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape))

decisionTree_train(X_train, Y_train,X_test, Y_test)
# clf = joblib.load('d:/tmp/decisionTree_train_alien.pkl') 
# PY[15] = clf.predict(PX)
# print PY
# logistic_train(X_train, Y_train,X_test, Y_test)
# poly_train(X_train, Y_train,X_test, Y_test)

# clf = svm.SVC(C=1.0,kernel='linear')
# clf.fit(X,Y) 

# plt.figure(figsize=(12,6), dpi=80, subplotpars=SubplotParams(hspace=0.3))#figure图像大小
# plt.subplot(1, 1, 1)
# scatter_ctgx(data_src, plt)
# plt.subplot(1, 2, 2)
# scatter_fire(data_src, plt)
# plt.show()


    