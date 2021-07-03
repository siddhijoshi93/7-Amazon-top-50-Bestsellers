# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:31:14 2021

@author: ADMIN
"""

import pandas as pd
import numpy as np

amazon=pd.read_csv("C:/Users/ADMIN/Desktop/Siddhi/bestsellers with categories.csv")

amazon.info()
amazon.describe()
amazon.head(5)
amazon.columns.nunique()
#finding missing values
amazon.isna().sum()
#checking for outliers
import matplotlib.pyplot as plt
plt.boxplot(amazon[['User Rating', 'Reviews', 'Price', 'Year']])
print(amazon.nunique())
amazon.corr()
import seaborn as sns
sns.heatmap(amazon.corr(), annot=True, cmap='coolwarm')
sns.pairplot(amazon)

#distributin of  genre vvisualization, check plot pie and countplot ftrom slack and google
amazon['Genre'].value_counts()
amazon['Genre'].value_counts().plot.pie(autopct="%.1f%%")

genre_peryear=amazon.groupby('Year')['Genre'].value_counts(ascending=False)
sns.countplot(amazon['Year'],hue= amazon['Genre'])
plt.title('Genre of books per year', fontsize=20)
plt.ylabel('Genre')

#understamding different ratings
amazon['User Rating'].mean()
amazon['User Rating'].max()
amazon['User Rating'].mode()
amazon['User Rating'].unique()
amazon['User Rating'].value_counts()

# distribution of ratings
total_ratings=amazon['User Rating'].value_counts(ascending=False)
sns.set(font_scale = 1.2, style='darkgrid')
sns.barplot(x=total_ratings.index, y=total_ratings.values, color='blue')
plt.title('Distribution of User Rating', fontsize=20)
plt.ylabel('Counts')
plt.xlabel('Ratings')

#books per author
a=dict(amazon['Author'].value_counts())
b=list(a.items())
print(b[:10])

#authors , amximumm ratings
maxrating= amazon[amazon['User Rating']==4.9]['Author'].value_counts(ascending=False)
sns.barplot(x=maxrating.index, y=maxrating.values, color='blue')
plt.title('Total highest rating per author')
plt.xticks(rotation=90)

print(amazon['Reviews'].max())
abs=amazon.groupby('Reviews')['Author'].value_counts()
print(amazon[amazon['Reviews']==amazon['Reviews'].max()])
#where the cradads sing by delia owens has the highst number of reviews i.e 87841

#price of books with highest ratings(check agai)
plt.figure(figsize=(12,6))
sns.distplot(amazon['Price'])
plt.title('Price Distribution Plot',fontsize=20)
amazon['Price'].mode()

#Model making
X=np.array(amazon[['Reviews',"Price", 'Year','User Rating']])
Y= np.array(amazon[['Genre']]).ravel()

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)

#training the model
clf.fit(X_train, y_train)

#prediction
pred=clf.predict(X_test)
actual_y=y_test

#accuracy
from sklearn import metrics
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(actual_y, pred))
#accuracy =0.781818181818181

#linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#predicting
y_pred= regressor.predict(X_test)
print(y_pred)

from sklearn import metrics
np.sqrt(metrics.mean_squared_error(actual_y,y_pred))
# 0.4714778305779233

regressor.score(X_test,y_test)
#0.10818127613056139









































#PREDICTING USER RATING OF A BOOK
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

LE=LabelEncoder()
LE.fit(amazon['Genre'])
amazon['Genre']=LE.transform(amazon['Genre'])
amazon.head()

col=['Reviews','Price','Year','Genre']
X=amazon.iloc[:,3:8].values
y=amazon[['User Rating']]

#splitting the data
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#using Linear regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
#training the model
model.fit(X_train,y_train)
#prediction
pred=model.predict(X_test)

#for accuracy
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
#accuracy=0.2295578451526144

#checking for random forrest
from sklearn.ensemble import RandomForestRegressor
m1 = RandomForestRegressor()
m1.fit(X_train, y_train)
prediction=m1.predict(X_test)

#accuracy
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

#gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor
m2 = GradientBoostingRegressor()
m2.fit(X_train, y_train)

#prediction
pred_y=m2.predict(X_test)
#accuracy
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_y)))

#decision tree
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=11).fit(X_train , y_train)

#prediction
preddy=dt.predict(X_test)
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, preddy)))
