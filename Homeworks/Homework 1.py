## The Perceptron Learning Algorithm ##

import numpy as np
import random
import matplotlib.pyplot as plt

N = 100
dim = 2

totalCounts = []
totalRuns = []
trials = 1000

for k in range(trials):
    linePoints = np.random.uniform(-1, 1, size=(2, dim))

    x1 = [linePoints[0][0], linePoints[1][0]]
    x2 = [linePoints[0][1], linePoints[1][1]]

    coefficients = np.polyfit(x1, x2, 1)
    a = coefficients[0]
    b = coefficients[1]

    initialPoints = np.random.uniform(-1, 1, N * 2)
    points = []

    for l in range(0, len(initialPoints), 2):
        points.append([initialPoints[l], initialPoints[l + 1], -1])

    xx = []
    yy = []

    for p in points:
        xx.append(p[0])
        yy.append(p[1])

    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(-1, 1, 100)
    y_axis = polynomial(x_axis)

    d = []

    for i in points:
        if i[1] - a * i[0] > b:
            d.append(1)
        else:
            d.append(-1)

    # numpy array conversion

    points = np.array(points)
    d = np.array(d)

    # plt.scatter(xx, yy, c=d)
    # plt.plot(x_axis, y_axis)
    # plt.show()

    w = np.zeros(len(points[0]))
    trialCounts = []
    runs = 0
    while True:
        count = 0
        indexes = []
        for index, j in enumerate(points):
            print(points[index], w)
            if np.sign(np.dot(points[index], w)) != d[index]:
                indexes.append(index)
                count += 1
            if len(indexes) != 0:
                ps = random.sample(indexes, 1)
                w += d[ps] * points[ps][0]
            else:
                continue
        trialCounts.append(count)
        runs += 1
        if count == 0:
            break
    totalCounts.append(np.average(trialCounts))
    totalRuns.append(runs)
    print(k)

countsAvgRatio = np.average(totalCounts) / N
runsAvg = np.average(totalRuns)

print('Probability of Disagreement: ', countsAvgRatio)
print('Average Runs: ', runsAvg)