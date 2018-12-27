#coding=utf-8 
'''
Created on 2018年4月6日

@author: bingqiw
'''
import numpy as np

n_dot = 200

X = np.linspace(-2*np.pi, 2*np.pi, 200)

# np.random.rand(）(0~1)之间的随机数
Y = np.sin(X) + 0.2 * np.random.rand(n_dot) - 0.1

X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
print X.shape, Y.shape
print "构造数据完成"

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

print "多项式拟合正选函数"
# 定义多项式模型
def polynomial_model(degree =1):
    polynomial_features = PolynomialFeatures(degree = degree, include_bias = False)
    linear_regression = LinearRegression(normalize = True)
    pipeline = Pipeline([("polynomial_features",polynomial_features),("linear_regression",linear_regression)])
    return pipeline

from sklearn.metrics import mean_squared_error
degrees = [2,3,5,10]
results = []

for d in degrees:
    model = polynomial_model(degree=d)
    model.fit(X,Y)
    train_score = model.score(X, Y)
    mse = mean_squared_error(Y, model.predict(X))
    results.append({"model":model,"degree":d,"score":train_score,"mse":mse})

for r in results:
    print("degree:{};train_score:{};mean squared error:{}".format(r["degree"],r["score"],r["mse"]))    

print"多项式拟合完成"

from matplotlib import pyplot as plt
from matplotlib.figure import SubplotParams

plt.figure(figsize=(12,6), dpi=80, subplotpars=SubplotParams(hspace=0.3))#figure图像大小

for i,r in enumerate(results):
    fig = plt.subplot(2, 2, i+1) #subplot设置子图的位置2*2，第i+1位置
    
    plt.xlim(-8,8)
    plt.title("LinearRegression degree={}".format(r["degree"]))
    plt.scatter(X, Y, s=2, c='b', alpha=0.5)
    plt.plot(X, r["model"].predict(X), 'r-')
    
plt.show()    
print "输出学习曲线"    