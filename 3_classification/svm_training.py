from sklearn import svm
# from sklearn.externals import joblib # if not Goolge Colab
import joblib # if Google Colab
import numpy as np

with open("iris_training.csv", 'r') as file:
    header = file.readline()
    data = np.loadtxt(file, delimiter=',', usecols=(0,1,2,3,4))

inputs = data[:,0:2]
inputs = inputs[(data[:,4] == 0) | (data[:,4] == 1)]
labels = data[:,4]
labels = labels[(labels == 0) | (labels == 1)]

type0 = inputs[labels==0]
type1 = inputs[labels==1]

training_inputs = np.r_[type0, type1]
training_labels = np.r_[np.zeros(len(type0)),np.ones(len(type1))]

clf = svm.SVC()
clf.fit(training_inputs, training_labels) 
joblib.dump(clf, 'svm.pkl') 
