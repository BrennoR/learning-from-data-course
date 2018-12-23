import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib import colors
import random

# train = pd.read_csv('6_train.txt', header=None, delim_whitespace=True)
# X_train = train.iloc[:, :2].values
# y_train = train.iloc[:, -1].values
#
# test = pd.read_csv('6_test.txt', header=None, delim_whitespace=True)
# X_test = test.iloc[:, :2].values
# y_test = test.iloc[:, -1].values
#
# X_train_transform = []
# X_test_transform = []
#
# for i in X_train:
#     X_train_transform.append([1, i[0], i[1], i[0] ** 2, i[1] ** 2, i[0] * i[1], np.abs(i[0] - i[1]),
#                               np.abs(i[0] + i[1])])
#
# for i in X_test:
#     X_test_transform.append([1, i[0], i[1], i[0] ** 2, i[1] ** 2, i[0] * i[1], np.abs(i[0] - i[1]),
#                               np.abs(i[0] + i[1])])
#
# X_train_transform = np.array(X_train_transform)
# X_test_transform = np.array(X_test_transform)
#
# for k in range(3,8):
#     X = X_train_transform[:, :k+1]
#     X_out = X_test_transform[:, :k+1]
#     idx_tr = np.random.choice(len(X), 25, replace=False)
#     # X_tr = X[idx_tr]
#     # y_tr = y_train[idx_tr]
#     # X_val = np.delete(X, idx_tr, axis=0)
#     # y_val = np.delete(y_train, idx_tr, axis=0)
#     X_val = X[0:25, :]
#     X_tr = X[25:, :]
#     y_val = y_train[0:25]
#     y_tr = y_train[25:]
#
#     w = np.zeros(X_tr.shape[1])
#     w_lin = np.dot(np.linalg.inv(np.transpose(X_tr).dot(X_tr)).dot(np.transpose(X_tr)), y_tr)
#
#     y_pred = np.sign(np.dot(X_val, w_lin))
#     print("{} - Accuracy: {}".format(k, accuracy_score(y_val, y_pred)))
#     y_pred_out = np.sign(np.dot(X_out, w_lin))
#     print("{} - Out of Sample - Accuracy: {}".format(k, accuracy_score(y_test, y_pred_out)))
#

# slope = (1/1.56)
# point = -1
# e = ((slope)*(point) - (1/1.56)) ** 2
# print(e)

# Problem 8

runs = 1000
N = 100
accuracies = []
num_vectors = []

for k in range(runs):
    while True:
        linePoints = np.random.uniform(-1, 1, size=(2, 2))
        x1 = [linePoints[0][0], linePoints[1][0]]
        x2 = [linePoints[0][1], linePoints[1][1]]

        coefficients = np.polyfit(x1, x2, 1)
        a = coefficients[0]
        b = coefficients[1]

        initialPoints = np.random.uniform(-1, 1, N * 2)
        X = []

        for l in range(0, len(initialPoints), 2):
            xx1 = initialPoints[l]
            xx2 = initialPoints[l + 1]
            X.append([xx1, xx2])

        X = np.array(X)
        X2 = X.copy()
        y = []

        for i in X:
            if i[1] - a * i[0] >= b:
                y.append(1)
            else:
                y.append(-1)

        if -1 in y and 1 in y:
            break

    target_f = np.poly1d(coefficients)

    x_axis = np.linspace(-1, 1, 100)
    y_axis = target_f(x_axis)

    # plt.scatter(X[:,0], X[:,1], c=y)
    # plt.plot(x_axis, y_axis)
    # plt.show()

    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    w = np.zeros(len(X[0]))

    while True:
        count = 0
        indexes = []
        for index, j in enumerate(X):
            if np.sign(np.dot(X[index], w)) != y[index]:
                indexes.append(index)
                count += 1
            if len(indexes) != 0:
                ps = random.sample(indexes, 1)
                w += np.dot(y[ps[0]], X[ps[0]])
            else:
                continue
        if count == 0:
            break

    newPoints = np.random.uniform(-1, 1, N * 2)
    new_X = []

    for l in range(0, len(newPoints), 2):
        xj1 = newPoints[l]
        xj2 = newPoints[l + 1]
        new_X.append([xj1, xj2])

    new_X = np.array(new_X)
    new_X2 = new_X.copy()
    new_X = np.concatenate((np.ones((new_X.shape[0], 1)), new_X), axis=1)
    new_y = []

    for i in new_X2:
        if i[1] - a * i[0] >= b:
            new_y.append(1)
        else:
            new_y.append(-1)

    # plt.scatter(X2[:, 0], X2[:, 1], c=y)
    # plt.plot(x_axis, y_axis)
    # plt.show()
    #
    # plt.scatter(new_X2[:, 0], new_X2[:, 1], c=new_y)
    # plt.plot(x_axis, y_axis)
    # plt.show()

    y_pred_percep = []
    for j in range(len(new_X)):
        if np.sign(np.dot(new_X[j], w)) > 0:
            y_pred_percep.append(1)
        else:
            y_pred_percep.append(-1)

    acc_percep = accuracy_score(new_y, y_pred_percep)

    svm = SVC(C=1e10, kernel='linear')
    svm.fit(X2, y)
    y_pred_svm = svm.predict(new_X2)

    acc_svm = accuracy_score(new_y, y_pred_svm)
    num_vectors.append(len(svm.support_vectors_))

    # # plot datapoints
    # plt.scatter(new_X2[:, 0], new_X2[:, 1], c=new_y, cmap=colors.ListedColormap(['blue', 'purple']))
    #
    # # plot decision boundaries
    # xmin, xmax = X2.min() - 1, X2.max() + 1
    # x1, x2 = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01))
    # xx = np.c_[x1.ravel(), x2.ravel()]
    # zz = svm.predict(xx)
    # zz = zz.reshape(x1.shape)
    # plt.contourf(x1, x2, zz, alpha=0.4, cmap=colors.ListedColormap(['blue', 'purple']))
    # plt.show()

    if acc_svm >= acc_percep:
        accuracies.append(1)
    else:
        accuracies.append(0)

print(np.average(num_vectors))
print(np.average(accuracies))
