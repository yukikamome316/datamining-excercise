import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

with open("out.csv", 'r') as file:
    line = file.readline()
    data = np.loadtxt(file, delimiter=',')

inputs = data[:,0]
inputs = np.array([[x,x**2,x**3,x**4,x**5,x**6,x**7,x**8] for x in inputs])

outputs = data[:,1]
regr = linear_model.LinearRegression()
regr.fit(inputs,outputs)
print('Coefficients: \n', regr.coef_)

parameters = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge = Ridge()
ridge_regressor = GridSearchCV(ridge,parameters, scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(inputs,outputs)

plt.scatter(inputs[:,0], outputs, color='black')
plot_x = np.arange(0,1,0.02)
test_x = np.array([[x,x**2,x**3,x**4,x**5,x**6,x**7,x**8] for x in plot_x])
pred = regr.predict(test_x)
plt.plot(plot_x, pred, color='blue', label='Linear Regression')
pred = ridge_regressor.predict(test_x)
plt.plot(plot_x, pred, color='red', label='Ridge Regression')
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-0.5, 1.5)
plt.legend()
plt.show()
