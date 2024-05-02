import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

with open("boston.csv", 'r') as file:
    line = file.readline()
    data = np.loadtxt(file, delimiter=',')

scaler = StandardScaler()
scaler.fit(data)

X = data[:, [0] + list(range(2, data.shape[1]))]
y = data[:,1]

regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X, y)

scores = cross_val_score(regr, X, y, cv=5, scoring='r2')
print('R2 score: ', scores)
