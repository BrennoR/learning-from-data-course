## Exercise 1.10 ##

import random
import matplotlib.pyplot as plt
import numpy as np

R = 1 # number of runs
n = 1000   # number of coins
N = 10     # times each coin is flipped individually

# 0 - tails, 1 - heads

allResults = []
results = []
total = 0
counter = 0

for a in range(R):
    for i in range(n):
        for j in range(N):
            randNum = random.randint(0,1)
            total += randNum
        freq = total/N
        results.append(freq)
        total = 0
    allResults.append(results)
    results = []
    counter+=1
    print(counter)

v1Array = []
vRandArray = []
vMinArray = []

for result in allResults:
    v1 = result[0]
    vRandNum = random.randint(0,n-1)
    vRand = result[vRandNum]
    vMin = min(result)
    v1Array.append(v1)
    vRandArray.append(vRand)
    vMinArray.append(vMin)



# plt.hist(v1Array, 11)
# plt.xlabel = 'Fraction of Heads'
# plt.ylabel = 'Frequency'
# plt.show()
#
# plt.hist(vRandArray, 10)
# plt.xlabel = 'Fraction of Heads'
# plt.ylabel = 'Frequency'
# plt.show()
#
# plt.hist(vMinArray, 10)
# plt.xlabel = 'Fraction of Heads'
# plt.ylabel = 'Frequency'
# plt.show()

epsilon = np.linspace(0.0,1.0,1000)
hoeff = 2*np.exp(-2*(epsilon**2)*(10))

plt.plot(epsilon, hoeff)
plt.show()