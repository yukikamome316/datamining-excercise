# from sklearn.externals import joblib # if not Golge Colab
import joblib # if Google Colab
import numpy as np

with open("iris_test.csv", 'r') as file:
    line = file.readline()
    data = np.loadtxt(file, delimiter=',', usecols=(0,1,2,3,4))

inputs = data[:,0:3]
inputs = inputs[(data[:,4] == 0) | (data[:,4] == 1)]
labels = data[:,4]
labels = labels[(labels == 0) | (labels == 1)]

clf = joblib.load('linear_svm.pkl')
results = clf.predict(inputs)

print("Answer : {0}".format(labels))
print("Predict: {0}".format(results))
