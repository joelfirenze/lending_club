
import pandas as pd

lending_df = pd.read_csv('loan.csv')


df = lending_df.drop(['id', 'member_id'], axis = 1)

df.corr().abs()
import numpy as np

import matplotlib.pyplot as plt


df['loan_status'].value_counts()

df['loan_status'].value_counts().plot(kind= 'bar')


plt.figure(figsize = (15,10))
plt.xticks(rotation = 45)
plt.bar(df['loan_status'].value_counts().index, df['loan_status'].value_counts())

# 8) Employment
plt.figure(figsize = (15,10))
plt.xticks(rotation = 45)
plt.bar(df['emp_length'].value_counts().index, df['emp_length'].value_counts())


import seaborn as sns

# 3) Loan amount distribtion (use distplot)
plt.figure(figsize = (15,10))
sns.distplot(df['loan_amnt'], bins = 10)

# 5) Instalment distribution (use distplot or barplot)
import matplotlib.pyplot as plt
plt.figure(figsize = (15,10))
sns.distplot(df['installment'], bins = 10)

# 2) Client purpose (use barplot)
import matplotlib.pyplot as plt

plt.figure(figsize = (15,10))
plt.xticks(rotation = 45)
df['purpose'].value_counts()

sns.barplot(x = df['purpose'].value_counts().index, y = df['purpose'].value_counts())

df["int_rate"].value_counts()
plt.figure(figsize = (15,10))
plt.xticks(rotation = 45)
sns.distplot(df['int_rate'], bins = 100)

df['grade']
plt.figure(figsize = (15,10))
plt.xticks(rotation = 45)
df['purpose'].value_counts()

sns.barplot(x = df['grade'].value_counts().index, y = df['grade'].value_counts())

plt.figure(figsize = (15,10))
plt.xticks(rotation = 45)
sns.scatterplot(x = df['grade'], y = df['int_rate'])

p1 = plt.bar(df['emp_length'].value_counts().index, df['emp_length'].value_counts())
p2 = plt.bar(df['int_rate'].value_counts().index, df['int_rate'].value_counts())
plt.xticks(rotation = 45)
plt.show()

#did a bunch of ther eda on the rest of the dataset

#and now i go to construct the model
columns = ['grade', 'emp_length', 'annual_inc', 'hardship_flag', 'loan_amnt', 'delinq_2yrs', 'installment', 'zip_code', 'tot_cur_bal', 'il_util', 'loan_status']

new_df['new_grade'] = new_df['grade']

#but first I have to change the code for some of the variables
new_df1 = new_df.replace({'new_grade':{'A':1, 'B':1, 'C':0, 'D':0, 'E':0, 'F':0, 'G':0}})

new_df1 = new_df1.replace({'emp_length':{'< 1 year':1, '1 year':1, '2 years':2, '3 years':3, '4 years': 4, '5 years': 5, '6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10}})

new_df1 = new_df1.replace({'loan_status':{'In Grace Period':1, 'Current':1, 'Fully Paid':1, 'Charged Off':0, 'Late (16-30 days)': 1, 'Late (31-120 days)':1, 'Does not meet the credit policy. Status:Fully Paid':1, 'Default':0, 'Does not meet the credit policy. Status:Charged Off':0}})


import re

df = pd.DataFrame()
df['emp_length'] = new_df1['emp_length']


df['annual_inc'] = new_df1['annual_inc']
df['loan_amnt'] = new_df1['loan_amnt']
df['delinq_2yrs'] = new_df1['delinq_2yrs']
df['installment'] = new_df1['installment']
df['tot_cur_bal'] = new_df1['tot_cur_bal']
df['il_util'] = new_df1['il_util']
df['new_grade'] = new_df1['new_grade']
df['loan_status'] = new_df1['loan_status']

df

df.fillna(0, inplace = True)
df


list(df.columns)


#really training the model now, first using linear regression
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
x = df[['emp_length','annual_inc','loan_amnt','delinq_2yrs','installment','tot_cur_bal','il_util','new_grade']]
y = df['loan_status']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


import numpy as np
from sklearn.linear_model import LinearRegression
lr = linear_model.LinearRegression()

model = lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)

print(lr.coef_)

from sklearn.metrics import mean_squared_error, r2_score

r2_score(y_test, y_predict)

#now using decision tree classifier

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)


y_pred_dt = dt.predict(x_test)


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_dt)


from sklearn.metrics import f1_score
f1_score(y_test, y_pred_dt)

#now using xgboost
!pip install xgboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error

data_dmatrix = xgb.DMatrix(data = x, label = y)

xg_class = xgb.XGBClassifier(objective = 'binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

xg_class.fit(x_train, y_train)

preds = xg_class.predict(x_test)


from sklearn import metrics
metrics.accuracy_score(y_test, preds)

#https://www.datacamp.com/community/tutorials/xgboost-in-python

from sklearn.metrics import f1_score
f1_score(y_test, preds)

#and now, random forest classifier
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 10)
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)


from sklearn.metrics import f1_score
f1_score(y_test, y_pred_rfc)

from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_rfc)

#and now, k nearest neighbours
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)


y_knn = model.predict(x_test)

metrics.accuracy_score(y_test, y_knn)
f1_score(y_test, y_knn)


#importance of the various features
from xgboost import plot_importance
plot_importance(xg_class)








