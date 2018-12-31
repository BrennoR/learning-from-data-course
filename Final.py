import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

# Problems 7 - 10

# # Dataset
# columns = ['digit', 'intensity', 'symmetry']
# train = pd.read_csv('8_train.txt', header=None, delim_whitespace=True, names=columns)
# test = pd.read_csv('8_test.txt', header=None, delim_whitespace=True, names=columns)
#
# y_train = train.iloc[:, 0]
# x_train = train.iloc[:, 1:]
#
# y_test = test.iloc[:, 0]
# x_test = test.iloc[:, 1:]
#
# digits_train = []
#
# for i in range(10):
#     digit_train = train[train.digit == i]
#     digit_train.digit = 1
#     digits_train.append(digit_train)
#
# digits_test = []
#
# for i in range(10):
#     digit_test = test[test.digit == i]
#     digit_test.digit = 1
#     digits_test.append(digit_test)
#
# versus_all_train = []
#
# for i in range(10):
#     digit_versus = train[train.digit != i]
#     digit_versus.digit = -1
#     versus_all_train.append(pd.concat([digits_train[i], digit_versus]))
#
# versus_all_test = []
#
# for i in range(10):
#     digit_versus = test[test.digit != i]
#     digit_versus.digit = -1
#     versus_all_test.append(pd.concat([digits_test[i], digit_versus]))
#
# # Versus All
# value = 1
#
# transformed_train = []
# for i in range(len(versus_all_train[value])):
#     x1 = versus_all_train[value].iloc[i, 1]
#     x2 = versus_all_train[value].iloc[i, 2]
#     label = versus_all_train[value].iloc[i, 0]
#     transformed_train.append([label, x1, x2, x1 * x2, x1 ** 2, x2 ** 2])
#
# transformed_train = pd.DataFrame(transformed_train)
#
# transformed_test = []
# for i in range(len(versus_all_test[value])):
#     z1 = versus_all_test[value].iloc[i, 1]
#     z2 = versus_all_test[value].iloc[i, 2]
#     z_label = versus_all_test[value].iloc[i, 0]
#     transformed_test.append([z_label, z1, z2, z1 * z2, z1 ** 2, z2 ** 2])
#
# transformed_test = pd.DataFrame(transformed_test)
#
# # ridge = RidgeClassifier(alpha=1)
# # ridge.fit(versus_all_train[value].iloc[:, 1:], versus_all_train[value].iloc[:, 0])
# # y_pred = ridge.predict(versus_all_test[value].iloc[:, 1:])
# # # y_in = ridge.predict(versus_all_train[value].iloc[:, 1:])
# # ridge.fit(transformed_train.iloc[:, 1:], transformed_train.iloc[:, 0])
# # y_pred_2 = ridge.predict(transformed_test.iloc[:, 1:])
# #
# # # print("E_in =", 1 - accuracy_score(y_in, versus_all_train[value].iloc[:, 0]))
# # print("E_out =", 1 - accuracy_score(y_pred, versus_all_test[value].iloc[:, 0]))
# # print("E_out_2 =", 1 - accuracy_score(y_pred_2, transformed_test.iloc[:, 0]))
#
# values = [1, 5]
#
# digits_train[values[1]].digit = -1
# digits_test[values[1]].digit = -1
# new_train = pd.concat([digits_train[values[0]], digits_train[values[1]]])
# j1 = new_train.iloc[:, 1]
# j2 = new_train.iloc[:, 2]
# final_train = pd.concat([new_train, j1 * j2, j1 ** 2, j2 ** 2], axis=1)
# new_test = pd.concat([digits_test[values[0]], digits_test[values[1]]])
# w1 = new_test.iloc[:, 1]
# w2 = new_test.iloc[:, 2]
# final_test = pd.concat([new_test, w1 * w2, w1 ** 2, w2 ** 2], axis=1)
#
# for a in [0.01, 1]:
#     ridge = RidgeClassifier(alpha=a)
#     ridge.fit(final_train.iloc[:, 1:], final_train.iloc[:, 0])
#     y_pred_in = ridge.predict(final_train.iloc[:, 1:])
#     y_pred_out = ridge.predict(final_test.iloc[:, 1:])
#     print("=" * 20, a, "=" * 20)
#     print("E_in:", 1 - accuracy_score(final_train.iloc[:, 0], y_pred_in))
#     print("E_out:", 1 - accuracy_score(final_test.iloc[:, 0], y_pred_out))
#
# # all_errors = []
# # E_in = []
# # E_out = []
# #
# # for i in range(100):
# #     errors = []
# #     errors_in = []
# #     for c in [0.01, 1, 100, 10e4, 10e6]:
# #         svm = SVC(kernel='rbf', degree=2, C=c)
# #         svm.fit(new_train.iloc[:, 1:], new_train.iloc[:, 0])
# #         y_pred = svm.predict(new_test.iloc[:, 1:])
# #         y_pred_in = svm.predict(new_train.iloc[:, 1:])
# #         # print("=" * 20, Q, c, "=" * 20)
# #         # print("Number of Support Vectors:", len(svm.support_vectors_))
# #         # print("E_out", 1 - accuracy_score(y_pred, new_test.iloc[:, 0]))
# #         errors.append(1 - accuracy_score(y_pred, new_test.iloc[:, 0]))
# #         errors_in.append(1 - accuracy_score(y_pred_in, new_train.iloc[:, 0]))
# #     all_errors.append(np.argmin(errors))
# #     E_out.append(np.argmin(errors))
# #     E_in.append(np.argmin(errors_in))
# #
# # print(np.average(all_errors))
# # print(np.average(E_out))
# # print(np.average(E_in))
#

# Problem 12

x = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
y = [-1, -1, -1, 1, 1, 1, 1]

clf = SVC(C=100000, kernel='poly', degree=2)
clf.fit(x, y)
print(len(clf.support_vectors_))
