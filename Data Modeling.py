#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:00:31 2020

@author: mahmoud
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




df = pd.read_csv('Cleaned_Data_EDA.csv')
#Shose relevant columns
df.columns

df_model=df[['Industry','Rating', 'Sector', 'Size', 'Type of ownership','City', 'State', 'Country', 'At_hq',
       'Company_age', 'Python', 'R', 'Spark', 'AWS', 'Excel', 'Hadoop',
       'Tableau', 'Power_bi', 'BI', 'Min_revenue', 'Max_revenue',
       'Average_revenue','Job Simplified', 'Job Senioriry', 'Desc Length',
       'Competitors Length','avg_salary']]
#Get dummy variables

df_dumm=pd.get_dummies(df_model)
#Train_test split
from sklearn.model_selection import train_test_split
X=df_dumm.drop('avg_salary',axis=1)
y=df_dumm['avg_salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Models building
#1 multiple linear regression
#1.1 Using stats models
import statsmodels.api as sm
X_sm= X = sm.add_constant(X)
mod = sm.OLS(y,X_sm)
res = mod.fit()
print(res.summary())
#1.2 Using sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lmod = LinearRegression()
lmod.fit(X_train,y_train)
scores = cross_val_score(lmod, X_train, y_train,scoring='neg_mean_absolute_error', cv=5)
np.mean(scores)
#2 Regularized linear model
#2.1 lasso regression
from sklearn.linear_model import Lasso
alpha=[]
error=[]
for i in range(1,100):
    alpha.append((i/100))
    lasso_m =Lasso(alpha=(i/100))
    scores = cross_val_score(lasso_m, X_train, y_train,scoring='neg_mean_absolute_error', cv=5)
    error.append(np.mean(scores))
plt.plot(alpha,error)
err=tuple(zip(alpha,error))
df_error=pd.DataFrame(err,columns=['alpha','error'])
df_error[df_error['error']==max(df_error['error'])]
lasso_m= Lasso(alpha=0.03)
lasso_m.fit(X_train,y_train)
scores = cross_val_score(lasso_m, X_train, y_train,scoring='neg_mean_absolute_error', cv=5)
np.mean(scores)
#2.2 Ridge regression
from sklearn.linear_model import Ridge
alpha=[]
error=[]
for i in range(1,100):
    alpha.append((i/10))
    ridge_m =Ridge(alpha=(i/10))
    scores = cross_val_score(ridge_m, X_train, y_train,scoring='neg_mean_absolute_error', cv=5)
    error.append(np.mean(scores))
plt.plot(alpha,error)
err=tuple(zip(alpha,error))
df_error=pd.DataFrame(err,columns=['alpha','error'])
df_error[df_error['error']==max(df_error['error'])]
ridge_m = Ridge(alpha=6.3)
ridge_m.fit(X_train,y_train)
scores = cross_val_score(ridge_m, X_train, y_train,scoring='neg_mean_absolute_error', cv=5)
np.mean(scores)
#3 SVM
from sklearn import svm
svm_m = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svm_m.fit(X_train,y_train)
scores = cross_val_score(svm_m, X_train, y_train,scoring='neg_mean_absolute_error', cv=5)
np.mean(scores)
#4 random forest
from sklearn.ensemble import RandomForestRegressor
rf_m = RandomForestRegressor()
rf_m.fit(X_train,y_train)
scores = cross_val_score(rf_m, X_train, y_train,scoring='neg_mean_absolute_error', cv=5)
np.mean(scores)
#5 XGBoost 
from sklearn import ensemble
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
xgb_m = ensemble.GradientBoostingRegressor(**params)
xgb_m.fit(X_train,y_train)
scores = cross_val_score(xgb_m, X_train, y_train,scoring='neg_mean_absolute_error', cv=5)
np.mean(scores)
#Tune models using GridSearch
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'),'max_depth':[3,4,5],'max_features':('auto','sqrt','log2')}
grid=GridSearchCV(rf_m,parameters,scoring='neg_mean_absolute_error',cv=5)
grid.fit(X_train,y_train)
grid.best_score_
grid.best_estimator_
# =============================================================================
#Best estimators for the grid search
# RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mae',
#                       max_depth=5, max_features='auto', max_leaf_nodes=None,
#                       max_samples=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=1,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       n_estimators=150, n_jobs=None, oob_score=False,
#                       random_state=None, verbose=0, warm_start=False)
# =============================================================================
#Test ensembls
from sklearn.metrics import mean_absolute_error
models=[lmod,lasso_m,ridge_m,svm_m,rf_m,xgb_m]
mdl=['lmod','lasso_m','ridge_m','svm_m','rf_m','xgb_m']
error=[]
for i in range(0,len(models)):
    tpred=models[i].predict(X_test)
    err=mean_absolute_error(y_test,tpred)
    error.append(err)
err=tuple(zip(mdl,error))
Models_err=pd.DataFrame(err,columns=['model','error'])
#Combine best two models
new_pred=(models[3].predict(X_test)+models[4].predict(X_test))/2
mean_absolute_error(y_test,new_pred)