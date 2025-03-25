import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import shapiro
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import explained_variance_score, r2_score, mean_absolute_error, mean_squared_error
import xgboost
from xgboost import XGBRegressor, plot_importance

from collections import Counter


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df=pd.read_csv(r'C:\Users\bhara\Downloads\auto-mpg.csv')
df.replace({'?':np.nan},inplace=True)

df['horsepower'] = pd.to_numeric(df['horsepower'])
df.drop(['car name'],axis=1,inplace=True)
X=df.drop("mpg",axis=1)
Y=df['mpg']
X

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=101)
X_train.head()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.transform(X_test)
X_train_std

xgbr = XGBRegressor()
xgb_params = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500,1000]}

gsXGB = GridSearchCV(xgbr, xgb_params, cv = 7, scoring='neg_mean_squared_error', 
                     refit=True, n_jobs = 5, verbose=True)
gsXGB.fit(X_train_std,Y_train)
XGB_best = gsXGB.best_estimator_
gsXGB.best_score_

ypred = XGB_best.predict(X_test_std)
print(ypred)

rmse=np.sqrt(mean_squared_error(Y_test,ypred))
print('RMSE: ',rmse)
print('R_square:', r2_score(Y_test,ypred))

list1=[[8,340,160,3609,8,70,1]]
list1=sc.transform(list1)
list1

prediction2=gsXGB.predict(list1)
prediction2

import pickle
with open('model_pic','wb') as f:
    pickle.dump(gsXGB,f)

with open('model_pic','rb') as f:
    mp=pickle.load(f)

list1=[[8,307,130,3504,12,70,1]]
list1=sc.transform(list1)
list1

mp.predict(list1)