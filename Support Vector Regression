
#Importing Libraries
import pandas as pd 
import numpy as np 
import  matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVR 
from sklearn.model_selection import train_test_split 

# Importing dataset
df = pd.read_csv('Position_Salaries.csv')
print(df.head())

# Assigning the dependent and independent variables

Y = df.iloc[: ,2].values
X = df.iloc[:,1:2].values

Y = Y.reshape(-1,1)
X = X.reshape(-1,1)

# Feature Scale 

sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
Y = sc_y.fit_transform(Y)

# Fit the SVR to the dataset 

svr_reg = SVR(kernel=  'rbf')
svr_reg.fit(X,Y)

# Visualising SVR result 
plt.scatter(X,Y, color = 'red')
plt.plot(X,svr_reg.predict(X), color = 'blue')
plt.xlabel('Postion Levels')
plt.ylabel('Salary')
plt.show()
