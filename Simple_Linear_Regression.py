#*****************************************************Simple Linear Regression**********************************************************************************************************

# Importing libraries
import pandas as  pd 
import numpy as np  
import matplotlib.pyplot as plt   
from sklearn.preprocessing import LabelEncoder,OneHotEncoder  
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import cross_validate  
import os
from sklearn.preprocessing import StandardScaler
print(os.getcwd())
# Importing dataset
df = pd.read_csv('Salary_Data.csv')
# Indexing using loc
X = df.loc[:,'YearsExperience'].values
Y = df.loc[:,'Salary'].values
#reshaping is done to make a 1d array to a 2d array
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)


#Splitting the data into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size = 0.7, random_state =0)
# Fitting Linear regression model to the training data and assigning predicted values to Y_pred
clf = LinearRegression()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

#Visualising training data 
plt.scatter(x= X,y = Y, color = 'red')
plt.scatter(x= X_train,y = clf.predict(X_train), color = 'blue')
plt.scatter(x= X_test, y = clf.predict(X_test), color = 'green')
plt.title('Visualising training data ')
plt.xlabel('Years of Experience ')
plt.ylabel('Salary')
plt.show()
