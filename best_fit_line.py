from statistics import mean 
import numpy as np
import matplotlib.pyplot as plt


xs = np.array([1,2,3,4,5,6])

ys = np.array([5,4,6,5,6,7])

def  best_fit_slope_intercept(xs,ys):
	m = ((mean(xs) * mean(ys) - mean(xs*ys)) /
		(mean(xs) * mean(xs) - mean(xs * xs))
		)
	b = mean(ys) - m * mean(xs)
	return m,b

def squared_error(ys_orig, ys_line):
	return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_regr = squared_error(ys_orig,ys_line)
	squared_error_y_mean = squared_error(ys_orig, y_mean_line)
	return 1- squared_error_regr /squared_error_y_mean



plt.scatter(xs,ys)
plt.show()
m,b = best_fit_slope_intercept(xs,ys)

print("The slope and intercept  of the best fit line is " + str(m)+"  "+str(b))

regression_line = [m*x +b for x in xs]
plt.scatter(xs, ys)
plt.plot(regression_line)
plt.show()

r_squared = coefficient_of_determination(ys, regression_line)
print("The rsquared values calculated was " + str(r_squared))
