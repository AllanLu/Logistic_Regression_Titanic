import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def loss(h, y):
    loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    return loss
 
def get_acc(x,y,w):
    ypred=sigmoid(np.dot(x,w))
    for i in range(len(ypred)):
        if ypred[i]>=0.5:
            ypred[i]=1
        else:
            ypred[i]=0
    return accuracy_score(y,ypred)

def Logistic_Regression(data,label,lr,iteration):
    intercept = np.ones((data.shape[0], 1)) 
    data = np.concatenate((intercept,data), axis=1) # 在左边加入一列 1
    xtrain,xval,ytrain,yval=train_test_split(data,label,test_size=0.1) #划分训练集和验证集
    weight = np.zeros(xtrain.shape[1]) # 初始化weight
    
    acc_val_max=0
    earlystop=0 # 想加入early stop，但是实际跑下来没用
    
    for i in range(iteration):
        h=sigmoid(np.dot(xtrain,weight))
        error=h-ytrain
        g=np.dot(xtrain.T,error)
        weight=weight-lr * g
        
        # 测试early stop
        if i > (iteration/10):
            acc=get_acc(xval,yval,weight)
            if acc_val_max>acc:
                earlystop+=1
                if earlystop>5:
                    print("earlystop at index:",i)
                    break
            else:
                earlystop=0
                acc_val_max=acc
    print("val acc:",acc_val_max)
    return weight

# 读入数据
data=pd.read_csv('titanic_train.csv')
data=data.loc[:,["Survived","Sex","Age"]]
data.replace('male',0,inplace=True)
data.replace('female',1,inplace=True)

# 对于缺失数据插值
imputer=Imputer()
data_i=np.array(imputer.fit_transform(data))
data_i=pd.DataFrame(data_i,columns=data.columns)
data=data_i.as_matrix()

# 分隔
X=data[:,1:3]
Y=data[:,0]

# 对年龄标准化
x1max,x2max=X.max(axis=0)
x1min,x2min=X.min(axis=0)
for i in X:
    i[1]=(i[1]-x2min)/(x2max-x2min)

# 划分训练集和测试集
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.1)

# 训练
weight=Logistic_Regression(xtrain, ytrain, 0.001, 5000)    # x特征矩阵，y目标值

# 预测
intercept = np.ones((xtest.shape[0], 1)) 
xtest = np.concatenate((intercept,xtest ), axis=1) # 在右边加入一列1
ypred=sigmoid(np.dot(xtest,weight))
for i in range(len(ypred)):
    if ypred[i]>=0.5:
        ypred[i]=1
    else:
        ypred[i]=0

# 评价
print("test acc:",accuracy_score(ytest,ypred))

# 参考代码
# # 梯度方向
# def gradient(X, h, y):
#     gradient = np.dot(X.T, (h - y)) / y.shape[0]
#     return gradient
 
# # 逻辑回归过程
# def Logistic_Regression1(x, y, lr=0.05, count=200):
#     intercept = np.ones((x.shape[0], 1)) # 初始化截距为 1
#     x = np.concatenate((intercept, x), axis=1)
#     w = np.zeros(x.shape[1]) # 初始化参数为 0
 
#     for i in range(count): # 梯度下降迭代
#         z = np.dot(x, w) # 线性函数
#         h = sigmoid(z)
 
#         g = gradient(x, h, y) # 计算梯度
#         w -= lr * g # 通过学习率 lr 计算步长并执行梯度下降
 
#         l = loss(h, y) # 计算损失函数值
 
#     z= z = np.dot(x, w)
#     h=sigmoid(z)
#     return l, w # 返回迭代后的梯度和参数