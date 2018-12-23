import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import integrate


# Problem 2

# d_vc = 50
# delta = 0.05
# N = np.linspace(0, 20000, 4001)
# sigmas_1 = np.sqrt((8/N)*np.log((4*2*N**d_vc)/delta))
# sigmas_2 = np.sqrt((2*np.log(2*N*N**d_vc))/N) + np.sqrt((2/N)*np.log(1/delta)) + 1/N
#
# # plt.plot(N, sigmas_1, N, sigmas_2)
# # plt.title('Bounds vs. Generalization Errors')
# # plt.xlabel('N')
# # plt.ylabel('Generalization Error')
# # plt.legend(['Original VC Bound', 'Rademacher Penalty Bound'])
#
# sigma, N_i = sp.symbols('sigma N_i')
# sigmas_3 = sp.Eq(sp.sqrt((1/N_i)*(2*sigma + sp.log((6*(2*N_i)**d_vc)/delta))), sigma)
# # sp.plot_implicit(sigmas_3, (N_i, 0, 20000), (sigma, 0, 1), title='Parrondo and Van den Broek')
#
# sigmas_4 = sp.Eq(sp.sqrt((1 / (2*N_i))*(4*sigma*(1 + sigma) + 4.38203 + 9.21034*50*2)), sigma)
# # sp.plot_implicit(sigmas_4, (N_i, 0, 20000), (sigma, 0, 1), title='Devroye')
# plt.show()
#
# print("Original VC Bound:", sigmas_1[2000])
# print("Rademacher Penalty Bound:", sigmas_2[2000])
# print("Parrondo and Van den Broek:", sp.solve(sigmas_3.subs(N_i, 10000), sigma)[0])
# print("Devroye:", sp.solve(sigmas_4.subs(N_i, 10000), sigma)[0])
#
# Problem 3

x_sin = np.linspace(-1, 1, 100)
y_sin = np.sin(np.pi*x_sin)

errors = []
a_list = []
b_list = []

for i in range(1000):
    x = np.random.uniform(-1, 1, 2)
    y = np.sin(np.pi * x)
    # coefficients = np.polyfit(x, y, 1)
    # a = coefficients[0]
    # b = coefficients[1]
    # a = (y[1] - y[0]) / 2 + y[0]
    a = sp.symbols('a')
    x_1 = x[0]
    y_1 = y[0]
    x_2 = x[1]
    y_2 = y[1]
    eq = sp.Eq(2 * (x_2 * (a * x_2 - y_2) + a * x_1 ** 2 - x_1 * y_1), 0)
    a = float(sp.solve(eq, a)[0])
    a_list.append(a)
    print(i)

a_hat = np.average(a_list)
# b_hat = b_list[index]
print("a_hat:", a_hat)
# print("b_hat:", b_hat)

variances = []

for i in range(1000):
    x = np.random.uniform(-1, 1, 2)
    temp_a = a_list[i]
    y = temp_a * x
    y_hat = a_hat * x
    mse2 = (y - y_hat) ** 2
    # mse2 = (mse2[0] + mse2[1]) / 2
    mse2 = np.average(mse2)
    variances.append(mse2)
    # yy = temp_a * x_sin
    # plt.plot(x_sin, x_sin)

y_pred = a_hat * x_sin

bias = np.average((y_sin - y_pred) ** 2)
print("bias", bias)

variance = np.average(variances)
print("variance", variance)

E_out = bias + variance
print("Expected E_out", E_out)

plt.plot(x_sin, y_sin, x_sin, y_pred)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.grid()
plt.show()
