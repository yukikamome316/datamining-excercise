import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

with open("boston.csv", 'r') as file:
    line = file.readline()
    data = np.loadtxt(file, delimiter=',')


inputs = data[:,:-2]
outputs = data[:,-1]

scaler = StandardScaler()
scaler.fit(inputs)

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, random_state=0)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

parameters = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge = Ridge()
ridge_regressor = GridSearchCV(ridge,parameters, scoring='neg_mean_squared_error', cv=5)
lasso = Lasso()
lasso_regressor = GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

ridge_regressor.fit(X_train,y_train)
lasso_regressor.fit(X_train,y_train)

ridge_pred = ridge_regressor.predict(X_test)
lasso_pred = lasso_regressor.predict(X_test)

print('Ridge R2 score: ', r2_score(y_test, ridge_pred))
print('Lasso R2 score: ', r2_score(y_test, lasso_pred))

plt.scatter(y_test, ridge_pred, color='blue', label='Ridge')
plt.scatter(y_test, lasso_pred, color='red', label='Lasso')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
