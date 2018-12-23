import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

train = pd.read_csv('6_train.txt', header=None, delim_whitespace=True)
X_train = train.iloc[:, :2].values
y_train = train.iloc[:, -1].values

test = pd.read_csv('6_test.txt', header=None, delim_whitespace=True)
X_test = test.iloc[:, :2].values
y_test = test.iloc[:, -1].values

X_train_transform = []
X_test_transform = []

for i in X_train:
    X_train_transform.append([1, i[0], i[1], i[0] ** 2, i[1] ** 2, i[0] * i[1], np.abs(i[0] - i[1]),
                              np.abs(i[0] + i[1])])

for i in X_test:
    X_test_transform.append([1, i[0], i[1], i[0] ** 2, i[1] ** 2, i[0] * i[1], np.abs(i[0] - i[1]),
                              np.abs(i[0] + i[1])])

X_train_transform = np.array(X_train_transform)
X_test_transform = np.array(X_test_transform)

k = -1
w = np.zeros(X_train_transform.shape[1])
lam = 10 ** k

w_lin = np.dot(np.linalg.inv(np.transpose(X_train_transform).dot(X_train_transform)).dot(np.transpose(X_train_transform)), y_train)
w_reg = np.dot(np.linalg.inv(np.transpose(X_train_transform).dot(X_train_transform) + np.dot(lam, np.identity(len(w)))).dot(np.transpose(X_train_transform)), y_train)

y_pred_train = np.sign(np.dot(X_train_transform, w_reg))
y_pred_test = np.sign(np.dot(X_test_transform, w_reg))

print("E_in:", 1 - accuracy_score(y_train, y_pred_train))
print("E_out:", 1 - accuracy_score(y_test, y_pred_test))

for l in range(1, 37):
    x = 8 * l + l * (36 - l)
    print("{} - {}".format(l, x))

