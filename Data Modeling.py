#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:00:31 2020

@author: mahmoud
"""

import pandas as pd
import numpy as np





df = pd.read_csv('DS_salary_cleaned.csv')
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
X=df_dumm.drop('avg_salary',axis=1)
y=df_dumm['avg_salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Models building
from sklearn.model_selection import train_test_split
#1 multiple linear regression
#1.1 Using stats models
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

X_sm= X = sm.add_constant(X)
mod = sm.OLS(y,X_sm)
res = mod.fit()
print(res.summary())
#1.2 Using sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lmod = LinearRegression()
scores = cross_val_score(lmod, X_train, y_train,scoring='neg_mean_absolute_error', cv=5)
np.mean(scores)
#2 Regularized linear model
#2.1 lasso regression
from sklearn.linear_model import Lasso
lasso_m= Lasso(alpha=0.1)
scores = cross_val_score(lasso_m, X_train, y_train,scoring='neg_mean_absolute_error', cv=5)
np.mean(scores)
#2.2 Ridge regression
#3 SVM
#4 random forest
#5 XGBoost 
#Tune models using GridSearch
#Test ensembls
