# importing libraries
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor

# loading dataset 

df = pd.read_csv('Position_Salaries.csv')

# Assigning dependent and independent variables 
# Remember iloc 
X = df.iloc[:,1:2].values
Y = df.iloc[:,2].values
# Reshaping the X and Y arrays
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)

# Fitting the decision tree regressor 

regressor = DecisionTreeRegressor()
regressor.fit(X,Y)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
