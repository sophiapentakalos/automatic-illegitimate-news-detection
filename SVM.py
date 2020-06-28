import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
import sys
import os

from utils import read_data, plot_data, plot_decision_function

# Read data
x, labels = read_data("fakeData.txt", "legitData.txt")

# Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size = 0.2, random_state=0)
'''
print("Displaying data. Close window to continue.")
print("X_train: {}".format(X_train))
# Plot data
plot_data(X_train, y_train, X_test, y_test)
'''
print("Training SVM with C=1 ...")
# make a classifier and fit on training data
clf_1 = svm.SVC(kernel='linear', C=1)
clf_1.fit(X_train, y_train)
'''
print("Display decision function (C=1) ...")
# Plot decision function on training and test data
plot_decision_function(X_train, y_train, X_test, y_test, clf_1)
'''
print("Training SVM with C=100 ...")
# make a classifier and fit on training data
clf_100 = svm.SVC(kernel='linear', C=100)
clf_100.fit(X_train, y_train)
train_acc = clf_100.fit(X_train, y_train).score(X_train, y_train)
print("Training loss clf-1: " + str(train_acc))
'''
print("Display decision function (C=100) ...")
# Plot decision function on training and test data
plot_decision_function(X_train, y_train, X_test, y_test, clf_100)
'''
clf_predictions = clf_1.predict(X_test)
print("Accuracy: {}%".format(clf_100.score(X_test, y_test) * 100 ))
