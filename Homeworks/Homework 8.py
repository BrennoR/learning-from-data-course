import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dataset
columns = ['digit', 'intensity', 'symmetry']
train = pd.read_csv('8_train.txt', header=None, delim_whitespace=True, names=columns)
test = pd.read_csv('8_test.txt', header=None, delim_whitespace=True, names=columns)

y_train = train.iloc[:, 0]
x_train = train.iloc[:, 1:]

y_test = test.iloc[:, 0]
x_test = test.iloc[:, 1:]

digits_train = []

for i in range(10):
    digit_train = train[train.digit == i]
    digit_train.digit = 1
    digits_train.append(digit_train)

digits_test = []

for i in range(10):
    digit_test = test[test.digit == i]
    digit_test.digit = 1
    digits_test.append(digit_test)

versus_all_train = []

for i in range(10):
    digit_versus = train[train.digit != i]
    digit_versus.digit = -1
    versus_all_train.append(pd.concat([digits_train[i], digit_versus]))

versus_all_test = []

for i in range(10):
    digit_versus = test[test.digit != i]
    digit_versus.digit = -1
    versus_all_test.append(pd.concat([digits_test[i], digit_versus]))

# Versus All
# value = 1
#
# svm = SVC(kernel='poly', degree=2, C=0.01)
# svm.fit(versus_all_train[value].iloc[:, 1:], versus_all_train[value].iloc[:, 0])
# y_pred = svm.predict(versus_all_test[value].iloc[:, 1:])
#
# print(len(svm.support_vectors_))
#
# print(1 - accuracy_score(y_pred, versus_all_test[value].iloc[:, 0]))

values = [1, 5]

digits_train[values[1]].digit = -1
digits_test[values[1]].digit = -1
new_train = pd.concat([digits_train[values[0]], digits_train[values[1]]])
new_test = pd.concat([digits_test[values[0]], digits_test[values[1]]])

all_errors = []
E_in = []
E_out = []

for i in range(100):
    errors = []
    errors_in = []
    for c in [0.01, 1, 100, 10e4, 10e6]:
        svm = SVC(kernel='rbf', degree=2, C=c)
        svm.fit(new_train.iloc[:, 1:], new_train.iloc[:, 0])
        y_pred = svm.predict(new_test.iloc[:, 1:])
        y_pred_in = svm.predict(new_train.iloc[:, 1:])
        # print("=" * 20, Q, c, "=" * 20)
        # print("Number of Support Vectors:", len(svm.support_vectors_))
        # print("E_out", 1 - accuracy_score(y_pred, new_test.iloc[:, 0]))
        errors.append(1 - accuracy_score(y_pred, new_test.iloc[:, 0]))
        errors_in.append(1 - accuracy_score(y_pred_in, new_train.iloc[:, 0]))
    all_errors.append(np.argmin(errors))
    E_out.append(np.argmin(errors))
    E_in.append(np.argmin(errors_in))

print(np.average(all_errors))
print(np.average(E_out))
print(np.average(E_in))

