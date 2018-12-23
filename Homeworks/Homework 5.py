import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Problem 1

# sigma = 0.1
# d = 8
# N = [10, 25, 100, 500, 1000]
#
# for n in N:
#     Exp_E_in = (sigma ** 2) * (1 - (d + 1)/n)
#     print(n, Exp_E_in)

# Problem 5

u, v = 1, 1
eta = 0.1

for i in range(16):
    E = (u * np.exp(v) - 2 * v * np.exp(-u)) ** 2
    print("{} - {}".format(i, E))
    E_grad_u = 2 * (u * np.exp(v) - 2 * v * np.exp(-u)) * (np.exp(v) + 2 * v * np.exp(-u))
    u -= eta * E_grad_u
    E_grad_v = 2 * (u * np.exp(v) - 2 * v * np.exp(-u)) * (u * np.exp(v) - 2 * np.exp(-u))
    # u -= eta*E_grad_u
    v -= eta*E_grad_v

print(u, v)
points = np.array([[1, 1], [0.713, 0.045], [0.016, 0.112], [-0.083, 0.029], [0.045, 0.024]])

for p in points:
    dist = np.linalg.norm(p - np.array([u, v]))
    print("{} - {}".format(p, dist))

# Problem 8
#
#
# def sigmoid(s):
#     value = 1 / (1 + np.exp(-s))
#     return value
#
# eta = 0.01
#
# N = 100
#
# accuracy_list = []
# error_list = []
# it_count_list = []
#
# for q in range(100):
#
#     linePoints = np.random.uniform(-1, 1, size=(2, 2))
#
#     x1 = [linePoints[0][0], linePoints[1][0]]
#     x2 = [linePoints[0][1], linePoints[1][1]]
#
#     coefficients = np.polyfit(x1, x2, 1)
#     a = coefficients[0]
#     b = coefficients[1]
#
#     initialPoints = np.random.uniform(-1, 1, N * 2)
#     points = []
#
#     for l in range(0, len(initialPoints), 2):
#         points.append([initialPoints[l], initialPoints[l + 1]])
#
#     d = []
#
#     for i in points:
#         if i[1] - a * i[0] > b:
#             d.append(1)
#         else:
#             d.append(-1)
#
#     # numpy array conversion
#
#     X = np.array(points)
#     y = np.array(d)
#
#     X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
#     w = np.zeros(X.shape[1])
#     w_minus = np.array([-1, -1, -1])
#
#     it_count = 0
#
#     while np.linalg.norm(w_minus - w) >= 0.01:
#         w_minus = w.copy()
#         k = np.random.choice(X.shape[0], 100)
#         for kk in k:
#             f = np.dot(X[kk], w)
#             grad = np.dot(y[kk], X[kk]) / (1 + np.exp(np.dot(y[kk], f)))
#             w += eta * grad
#         it_count += 1
#
#     it_count_list.append(it_count)
#     testPoints = np.random.uniform(-1, 1, N * 2)
#     points2 = []
#
#     for l in range(0, len(testPoints), 2):
#         points2.append([testPoints[l], testPoints[l + 1]])
#
#     d2 = []
#
#     for i in points2:
#         if i[1] - a * i[0] > b:
#             d2.append(1)
#         else:
#             d2.append(-1)
#
#     d2 = np.array(d2)
#
#     points2 = np.concatenate((np.ones((len(points2), 1)), points2), axis=1)
#
#     y_hat = []
#     for l in range(len(points2)):
#         if np.dot(points2[l], w) > 0:
#             y_hat.append(1)
#         else:
#             y_hat.append(0)
#
#
#     accuracy = accuracy_score(d2, y_hat)
#     accuracy_list.append(accuracy)
#     error = 0
#     for n in range(len(d2)):
#         error += np.log(1 + np.exp(np.dot(-d2[n], np.dot(points2[n], w))))
#     error /= len(d2)
#     error_list.append(error)
#
# print("E_out average: ", np.average(error_list))
# print("Average number of iterations: ", np.average(it_count_list))
