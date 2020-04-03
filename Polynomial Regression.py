# importing libraries
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
df  = pd.read_csv('Position_Salaries.csv')
print(df.head())
Y = df.iloc[:,2].values
X= df.iloc[:,1:2].values

# Splitting train test 
X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state= 0)

# Fitting the model 
poly_reg = PolynomialFeatures(degree= 4 )
X_Poly = poly_reg.fit_transform(X)
print(X)
print(X_Poly)
poly_reg.fit(X_Poly,Y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_Poly, Y)
# Visualising the results 

plt.scatter(X,Y)
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()
