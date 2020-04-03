importing libraries
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler 
import os 
from sklearn.preprocessing import OneHotEncoder,LabelEncoder 
from sklearn.compose import ColumnTransformer 



# importing the dataset 
df = pd.read_csv('50_Startups.csv')
print(df.head())

# Assigning dependent and independent variables
Y =  df.loc[:,'Profit'].values

X = df.iloc[:,:-1].values

# Encoding the categorical variables 
from sklearn.compose import ColumnTransformer, make_column_transformer
preprocessor = make_column_transformer( (OneHotEncoder(),[3]),remainder="passthrough")
X = preprocessor.fit_transform(X)
print(type(X))
X = X[:,1:] # avoiding dummy variable trap 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0) 
regressor = LinearRegression()
regressor.fit(X_train,Y_train) 
Y_pred = regressor.predict(X_test)
print(Y_pred)
