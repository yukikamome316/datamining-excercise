import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

with open("boston.csv", 'r') as file:
    line = file.readline()
    data = np.loadtxt(file, delimiter=',')

scaler = StandardScaler()
scaler.fit(data)

inputs = data[:, [0] + list(range(2, data.shape[1]))]
outputs = data[:,1]

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, random_state=42)

regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)

print('Coefficients: \n', regr.coef_)

pred = regr.predict(X_test)

print('R2 score: ', r2_score(y_test, pred))
