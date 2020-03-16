# Importing libraries
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt 

#Loading the dataset
df = pd.read_csv('Position_Salaries.csv')

# assigning dependent and independent variables 
Y  = df.iloc[ : , 2].values
X = df.iloc[ : ,1:2].values

# Fitting random forest regression to the dataset 

reg = RandomForestRegressor(n_estimators=10, random_state= 0)
reg.fit(X,Y) 

# Visualising random forest regression result 
X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.title("Truth or bluff(Random Forest Regression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
