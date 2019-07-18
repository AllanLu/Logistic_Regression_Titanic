import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score,recall_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

# boston=load_boston()
# xtrain,xtest,ytrain,ytest=train_test_split(boston.data,boston.target)

# model=xgb.XGBRegressor()
# model.fit(xtrain,ytrain)
# ypred=model.predict(xtest)
# print(r2_score(ytest,ypred))

# data = pd.read_csv("bmi.csv")
# X=np.array(data[['height','weight']])
# Y=np.array(data['label'])
# xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.1)
# clf=xgb.XGBClassifier()
# clf.fit(xtrain,ytrain)
# ypred=clf.predict(xtest)
# print(accuracy_score(ytest,ypred))
# print(recall_score(ytest,ypred,average='macro'))


data=pd.read_csv('titanic_train.csv')
#data=data.loc[:,["Survived","Sex","Age","Pclass"]] 0.8035
data=data.loc[:,["Survived","Sex","Age"]] #0.7978
data.replace('male',0,inplace=True)
data.replace('female',1,inplace=True)

imputer=Imputer()
data_i=np.array(imputer.fit_transform(data))
data_i=pd.DataFrame(data_i,columns=data.columns)
data=data_i.as_matrix()

X=data[:,1:]
Y=data[:,0]
sumacc=0
for i in range(1000):
    xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.1)
    model=xgb.XGBClassifier()
    model.fit(xtrain,ytrain)
    ypred=model.predict(xtest)
    sumacc+=accuracy_score(ytest,ypred)
    #print(accuracy_score(ytest,ypred))
print("total:",sumacc/1000)
    

