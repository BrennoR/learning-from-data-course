import random
import numpy as np
import matplotlib.pyplot as plt

# HOEFFDING INEQUALITY

# nu_1_list = []
# nu_min_list = []
# nu_rand_list = []
#
# for run in range(100000):
#     coin_list = []
#     for i in range(1000):
#         count = 0
#         for j in range(10):
#             if random.randint(0, 1) == 1:
#                 count += 1
#         coin_list.append(count)
#
#     c1 = coin_list[0]
#     nu_1 = c1 / 10
#     nu_1_list.append(nu_1)
#
#     c_min = coin_list[np.argmin(coin_list)]
#     nu_min = c_min / 10
#     nu_min_list.append(nu_min)
#
#     c_rand = coin_list[random.randint(0, 999)]
#     nu_rand = c_rand / 10
#     nu_rand_list.append(nu_rand)
#
#     print(run)
#
# print('Average Value of nu_1: {}'.format(np.average(nu_1_list)))
# print('Average Value of nu_min: {}'.format(np.average(nu_min_list)))
# print('Average Value of nu_rand: {}'.format(np.average(nu_rand_list)))
#
# plt.hist(nu_1_list)
# plt.title('nu_1')
# plt.xlabel('Fraction of Heads')
# plt.ylabel('Frequency')
# plt.show()
#
# plt.hist(nu_min_list)
# plt.title('nu_min')
# plt.xlabel('Fraction of Heads')
# plt.ylabel('Frequency')
# plt.show()
#
# plt.hist(nu_rand_list)
# plt.title('nu_rand')
# plt.xlabel('Fraction of Heads')
# plt.ylabel('Frequency')
# plt.show()

# LINEAR REGRESSION

# N = 10
# dim = 2
#
# E_in_list = []
# g_list = []
# a_list = []
# b_list = []
# total_runs = []
#
# for run in range(5000):
#
#     linePoints = np.random.uniform(-1, 1, size=(2, dim))
#
#     x1 = [linePoints[0][0], linePoints[1][0]]
#     x2 = [linePoints[0][1], linePoints[1][1]]
#
#     coefficients = np.polyfit(x1, x2, 1)
#     a = coefficients[0]
#     a_list.append(a)
#     b = coefficients[1]
#     b_list.append(b)
#
#     initialPoints = np.random.uniform(-1, 1, N * 2)
#     points = []
#
#     for l in range(0, len(initialPoints), 2):
#         points.append([1, initialPoints[l], initialPoints[l + 1]])
#
#     x1 = []
#     x2 = []
#
#     for p in points:
#         x1.append(p[1])
#         x2.append(p[2])
#
#     y = []
#
#     for i in points:
#         if i[2] - a * i[1] > b:
#             y.append(1)
#         else:
#             y.append(-1)
#
#     target_f = np.poly1d(coefficients)
#
#     x_axis = np.linspace(-1, 1, 100)
#     y_axis = target_f(x_axis)
#
#     # plt.scatter(x1, x2, c=y)
#     # plt.plot(x_axis, y_axis)
#     # plt.show()
#
#     w = np.zeros(len(points[0]))
#
#     X = np.array(points)
#     y = np.array(y)
#
#     # w = np.dot(np.dot(np.matrix(np.dot(X.transpose(), X)).I, X.transpose()), y)
#     w = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)
#     g_list.append(w)
#
#     count = 0
#
#     for j in range(len(X)):
#         if (np.dot(X[j], w.transpose()) * y[j]) <= 0:
#             count += 1
#
#     E_in = count / 100
#     E_in_list.append(E_in)
#
#     # perceptron implementation
#
#     runs = 0
#
#     while True:
#         count = 0
#         indexes = []
#         for index, j in enumerate(X):
#             if np.sign(np.dot(X[index], w)) != y[index]:
#                 indexes.append(index)
#                 count += 1
#             if len(indexes) != 0:
#                 ps = random.sample(indexes, 1)
#                 # w += y[ps] * X[ps][0]
#                 w += np.dot(y[ps], X[ps])
#             else:
#                 continue
#         runs += 1
#         if count == 0:
#             break
#     total_runs.append(runs)
#     print(run)
#
# E_in_avg = np.average(E_in_list)
# print("E_in_avg:", E_in_avg)
# g_list = np.array(g_list)
# E_out_list = []

# for run in range(1000):
#
#     initialPoints = np.random.uniform(-1, 1, 1000 * 2)
#     points = []
#
#     for l in range(0, len(initialPoints), 2):
#         points.append([1, initialPoints[l], initialPoints[l + 1]])
#
#     y = []
#
#     for i in range(len(points)):
#         if points[i][2] - a_list[i] * points[i][1] > b_list[i]:
#             y.append(1)
#         else:
#             y.append(-1)
#
#     X = np.array(points)
#     y = np.array(y)
#
#     count = 0
#
#     for j in range(len(X)):
#         if (np.dot(g_list[j].transpose(), X[j]) * y[j]) <= 0:
#             count += 1
#
#     E_out = count / 1000
#     E_out_list.append(E_out)
#
# E_out_avg = np.average(E_out_list)
# print("E_out_avg:", E_out_avg)

# perceptron implementation results

# avg_runs = np.average(total_runs)
# print("Avg Perceptron Runs:", avg_runs)

# NONLINEAR TRANSFORMATION

N = 1000
E_in_list = []
c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
c6 = 0

runs = 100
h_list = []

for run in range(runs):

    initialPoints = np.random.uniform(-1, 1, N * 2)
    points = []
    nlpoints = []

    for l in range(0, len(initialPoints), 2):
        points.append([1, initialPoints[l], initialPoints[l + 1]])

    for l in range(0, len(initialPoints), 2):
        x1 = initialPoints[l]
        x2 = initialPoints[l + 1]
        nlpoints.append([1, x1, x2, x1*x2, x1 ** 2, x2 ** 2])

    noise = random.sample(list(np.arange(N)), 100)

    for n in noise:
        points[n] = np.negative(points[n])
        nlpoints[n] = np.negative(nlpoints[n])

    # X = np.array(points)
    X = np.array(nlpoints)

    y = []

    for p in X:
        yy = np.sign(p[1] ** 2 + p[2] ** 2 - 0.6)
        y.append(yy)

    w = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)

    c1 += w[0]
    c2 += w[1]
    c3 += w[2]
    c4 += w[3]
    c5 += w[4]
    c6 += w[5]

    count = 0
    h1 = 0
    h2 = 0
    h3 = 0
    h4 = 0
    h5 = 0

    for j in range(len(X)):
        x1 = X[j][1]
        x2 = X[j][2]
        if (np.dot(X[j], w.transpose()) * y[j]) <= 0:
            count += 1
        if np.sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*x1 ** 2 + 1.5*x2 ** 2) * y[j] <= 0:
            h1 += 1
        if np.sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*x1 ** 2 + 15*x2 ** 2) * y[j] <= 0:
            h2 += 1
        if np.sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*x1 ** 2 + 15*x2 ** 2) * y[j] <= 0:
            h3 += 1
        if np.sign(-1 - 1.5*x1 + 0.08*x2 + 0.13*x1*x2 + 0.05*x1 ** 2 + 0.05*x2 ** 2) * y[j] <= 0:
            h4 += 1
        if np.sign(-1 - 0.05*x1 + 0.08*x2 + 1.5*x1*x2 + 0.15*x1 ** 2 + 0.15*x2 ** 2) * y[j] <= 0:
            h5 += 1

    h1 /= N
    h2 /= N
    h3 /= N
    h4 /= N
    h5 /= N

    E_in = count / N
    E_in_list.append(E_in)
    h_list.append((h1, h2, h3, h4, h5))

h1_t = 0
h2_t = 0
h3_t = 0
h4_t = 0
h5_t = 0

for i in h_list:
    h1_t += i[0]
    h2_t += i[1]
    h3_t += i[2]
    h4_t += i[3]
    h5_t += i[4]

E_in_avg = np.average(E_in_list)
print("E_in_avg:", E_in_avg)

print(c1 / runs, c2 / runs, c3 / runs, c4 / runs, c5 / runs, c6 / runs)
print(h1_t / runs, h2_t / runs, h3_t / runs, h4_t / runs, h5_t / runs)

## E_out

E_out_list = []

for run in range(runs):
    initialPoints = np.random.uniform(-1, 1, N * 2)
    nlpoints = []

    for l in range(0, len(initialPoints), 2):
        x1 = initialPoints[l]
        x2 = initialPoints[l + 1]
        nlpoints.append([1, x1, x2, x1*x2, x1 ** 2, x2 ** 2])

    noise = random.sample(list(np.arange(1000)), 100)

    for n in noise:
        nlpoints[n] = np.negative(nlpoints[n])

    X = np.array(nlpoints)

    y = []

    for p in X:
        yy = np.sign(p[1] ** 2 + p[2] ** 2 - 0.6)
        y.append(yy)

    count = 0

    for j in range(len(X)):
        x1 = X[j][1]
        x2 = X[j][2]
        if np.sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*x1 ** 2 + 1.5*x2 ** 2) * y[j] <= 0:
            count += 1

    E_out = count / 1000
    E_out_list.append(E_out)

E_out_avg = np.average(E_out_list)
print("E_out_avg", E_out_avg)