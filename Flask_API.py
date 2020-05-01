#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:40:33 2020

@author: mahmoud
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:00:31 2020

@author: mahmoud
"""

import pandas as pd
df = pd.read_csv('Cleaned_Data_EDA.csv')
#Choose relevant columns
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
#Combined Model
from sklearn.ensemble import VotingRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
svm_m = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svm_f=svm_m.fit(X_train,y_train)
#score_svm=svm_f.score(X_test,y_test)
rf_m = RandomForestRegressor()
rf_f=rf_m.fit(X_train,y_train)
#score_rf=rf_f.score(X_test,y_test)
model_Ensemble = VotingRegressor(estimators=[('SVM', svm_m), ('RF', rf_m)],weights=[1, 1])
#model_Ensemble_f=
model_Ensemble.fit(X_train,y_train)
#score_Ensemble=model_Ensemble_f.score(X_test,y_test)
#prediction=
model_Ensemble.predict(X_test)
#mean_absolute_error(y_test,prediction)   


# =============================================================================
# #Pickling converts the object into a byte stream which can be stored,
# #transferred, and converted back to the original model at a later time.
# #Pickles are one of the ways python lets you save just about any object out of the box.
# =============================================================================
import pickle
pickl = {'model': model_Ensemble}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )


    
