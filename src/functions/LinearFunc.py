#coding=utf-8 
'''
Created on 2018年4月6日

@author: bingqiw
'''
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import pandas as pd
from Tkconstants import OFF

def func_to_dist(item, index1, index2):
    vec1 = item[index1]
    vec2 = item[index2]
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))  
    return dist

def func_to_ctgx(item):
    b_sx = item[11]
    a_sx = item[4]
#     
    b_sy = item[12]
    a_sy = item[5]
    
    ctgx = (b_sy - a_sy) / (b_sx - a_sx)
    return ctgx

def func_to_logisc(item):
#     val = np.fabs(item[13])
    val = item[13]
    offset = 0.1
    result = np.where((val>-(-2.0 - offset) and val<-(-2.0 + offset)) or (val>0.9 and val<1.1),1,0)
    return result  

def scatter_dist(data_src):
    data_src = pd.DataFrame(data_src).sort_values(by=13) 
#     print data_src.loc[:,[4,5,9,11,12,13,14]]
    X = data_src.iloc[0:100,13] #
    Y = data_src.iloc[0:100,14] #
    num = range(100)
    plt.scatter(num,X, s=50, c='b', alpha=0.5)
    plt.scatter(num,Y, s=50, c='r', alpha=0.5)
    plt.show()
    
def scatter_fire(df, plt):
    X = df.loc[(df[14] == 1)][4,5,6,7]
    Y = df.loc[df[14] == 0][13]
    plt.scatter(range(len(X)),X, s=100, c='b', alpha=0.5, marker='^')
#     plt.scatter(len(Y),Y, s=100, c='r', alpha=0.5, marker='v')
    
def scatter_ctgx(data_src, plt):    
    X = data_src.iloc[:,13] #
    Y = data_src.iloc[:,14] #
    num = range(X.shape[0])
    plt.scatter(num,X, s=50, c='b', alpha=0.5)
#     plt.scatter(num,Y, s=50, c='r', alpha=0.5)
    plt.ylim(-4,4)
      
def polynomial_model(degree =1):
    polynomial_features = PolynomialFeatures(degree = degree, include_bias = False)
    linear_regression = LinearRegression(normalize = True)
    pipeline = Pipeline([("polynomial_features",polynomial_features),("linear_regression",linear_regression)])
    return pipeline