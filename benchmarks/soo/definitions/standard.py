import math
import numpy as np
from functools import partial

from benchmarks.soo.optima import optima

'''
Input: scalari, liste che rappresentano un punto D dimensionale, 
    np.array monodimensionali che rappresentano un punto D dimensionle,
    np.array [D, N] che rappresentano N di punti D dimensionali (righe: dimensioni, colonne: punti)
    lista di lista gestita come numpy array [D, N]
'''


# Ackley Function 1 (Back and Schwefel, 1993) (continuous, differentiable, non-separable, scalable, multimodal)
def ackley_ext(x, a=20, b=0.2, c=2 * math.pi):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        # Avoid mono-dimensional points.
        x = x[:, np.newaxis]
    # At this point, x is a matrix where each column is an individual and each row a dimension.
    y = - a * np.exp(-b * np.sqrt(np.mean(x ** 2, axis=0))) - np.exp(np.mean(np.cos(c * x), axis=0)) + a + np.e
    # The axis=0 operation parallelizes the sum of the matrix directly using an efficient NumPy operation.
    if y.ndim == 1:
        # If the input is a scalar then return a scalar,
        y = np.squeeze(y)
    return y


ackley = partial(ackley_ext, a=20, b=0.2, c=2 * math.pi)


def adjiman(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x1, x2 = x
    y = np.cos(x1) * np.sin(x2) - x1 / (x2 ** 2 + 1)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


# Alpine Function 1 (Rahnamyan et al., 2007a) (continuous, non-differentiable, separable, non-scalable, multimodal)
def alpine01(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = np.sum(np.absolute(x * np.sin(x) + .1 * x), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def alpine02(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = np.prod(np.sqrt(x) * np.sin(x), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


# Bohachevsky Function 1 (Bohachevsky et al., 1986) (continuous, differentiable, separable, non-scalable, multimodal)
def bohachevsky(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x_curr = x[:-1, :]
    x_next = x[1:, :]
    y = np.sum(x_curr ** 2 + 2 * x_next ** 2
               - .3 * np.cos(3 * np.pi * x_curr) - .4 * np.cos(4 * np.pi * x_next) + .7, axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


# The sixth Bukin function has many local minima, all of which lie in a ridge.
def bukin06(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x1, x2 = x
    y = 100 * np.sqrt(np.absolute(x2 - .01 * x1 ** 2)) + .01 * np.absolute(x1 + 10)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def cross_in_tray(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x1, x2 = x
    y = -.0001 * (np.absolute(np.sin(x1) * np.sin(x2) *
                              np.exp(np.absolute(100 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))) + 1) ** .1
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def damavandi(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    factor_1 = (1 - np.absolute(
        (np.prod(np.sin(np.pi * (x - 2)), axis=0)) / (np.pi ** 2 * np.prod((x - 2), axis=0))
    ) ** 5)
    factor_2 = 2
    for i in range(len(x)):
        factor_2 += 2 ** i * (x[i] - 7) ** 2
    y = factor_1 * factor_2
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def deceptive(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    # The optimum is located in the values of the alpha, here alpha were randomly extracted froma normal distribution
    alphas = optima['Deceptive']
    alpha = np.array(alphas[: len(x)])
    alpha = alpha[:, np.newaxis, np.newaxis] if x.ndim == 3 else alpha[:, np.newaxis]
    beta = 2

    def gi(xi, ai):
        yi = (xi - 1) / (1 - ai)
        yi = np.where((0 <= xi) & (xi <= 4 / 5 * ai), - xi / ai + 4 / 5, yi)
        yi = np.where((4 / 5 * ai < xi) & (xi <= ai), 5 * xi / ai - 4, yi)
        yi = np.where((ai < xi) & (xi <= (1 + 4 * ai) / 5), 5 * (xi - ai) / (ai - 1), yi)
        return yi

    y = gi(x, alpha)
    z = - np.mean(y, axis=0) ** beta
    if z.ndim == 1:
        z = np.squeeze(z)
    return z


def de_villiers_glasser02(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x1, x2, x3, x4, x5 = x
    y = np.zeros(len(x))
    for i in range(1, 25):
        t_i = .1 * (i - 1)
        y_i = 53.81 * (1.27 ** t_i) * np.tanh(3.012 * t_i + np.sin(2.13 * t_i)) * np.cos(np.e ** .507 * t_i)
        y += (x1 * x2 ** t_i * np.tanh(x3 * t_i + np.sin(x4 * t_i)) * np.cos(t_i * np.e ** x5) - y_i) ** 2
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def drop_wave(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x1, x2 = x
    y = - (1 + np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2))) / (.5 * (x1 ** 2 + x2 ** 2) + 2)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def egg_holder(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x1, x2 = x
    y = - (x2 + 47) * np.sin(np.sqrt(np.absolute(x2 + x1 / 2 + 47))) - x1 * np.sin(
        np.sqrt(np.absolute(x1 - (x2 + 47))))
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def ferretti(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = 30.0 + np.sum(np.absolute(x), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


# Griewank Function (Griewank, 1981) (continuous, differentiable, non-separable, scalable, multimodal)
def griewank(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    i = np.empty(x.shape)
    for j in range(len(i)):
        i[j] = j + 1
    y = np.sum(x ** 2 / 4000, axis=0) \
        - np.prod(np.cos(x / np.sqrt(i)), axis=0) \
        + 1
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def holder_table(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x1, x2 = x
    y = - np.absolute(np.sin(x1) * np.cos(x2) * np.exp(np.absolute(1 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi)))
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def michalewicz(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    m = 10
    i = np.empty(x.shape)
    for j in range(len(i)):
        i[j] = j + 1
    y = - np.sum(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** (2 * m), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def mishra01(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    xn = len(x) - np.sum(x[:-1], axis=0)
    y = (1 + xn) ** xn
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def nobile1(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n = len(x)
    sum1 = np.sum(1 / (.001 + np.exp(x - 10 ** -n)), axis=0)
    sum2 = np.sum((x - 10 ** -n) ** 2, axis=0)
    y = np.sin(sum1) - 1 / (.001 + np.sqrt(sum2))
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def nobile2(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    i = np.empty(x.shape)
    for j in range(len(i)):
        i[j, :] = j + 1
    y = np.sum(x ** (i / (2 * len(x))), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def nobile3(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    i = np.empty(x.shape)
    for j in range(len(i)):
        i[j] = j + 1
    sum1 = np.sum(1 / (.001 + np.exp(x - 10 ** -i)), axis=0)
    sum2 = np.sum((x - 10 ** -i) ** 2, axis=0)
    y = np.sin(sum1) - 1 / (.001 + np.sqrt(sum2))
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def plateau(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = np.sum(np.floor(x), axis=0) + 30
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


# Quintic Function (Mishra, 2006f) (continuous, differentiable, separable, non-scalable, multimodal)
def quintic(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = np.sum(np.fabs(x ** 5 + 3 * x ** 4 + 4 * x ** 3 + 2 * x ** 2 - 10 * x - 4), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def rastrigin_ext(x, a=10.0):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = a * len(x) + np.sum(x ** 2 - a * np.cos(2 * np.pi * x), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


rastrigin = partial(rastrigin_ext, a=10.0)


def ripple01(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = np.sum(-np.e ** (-2 * np.log(2 * ((x - .1) / .8) ** 2))
               * (np.sin(5 * np.pi * x) ** 6 + .1 * np.cos(500 * np.pi * x) ** 2), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def rosenbrock(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x_curr = x[:-1, :]
    x_next = x[1:, :]
    y = np.sum(100 * (x_next - x_curr ** 2) ** 2 + (x_curr - 1) ** 2, axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def salomon(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
        x = x[:, np.newaxis]
    y = 1 - np.cos(2 * np.pi * np.sqrt(np.sum(x ** 2, axis=0))) + .1 * np.sqrt(np.sum(x ** 2, axis=0))
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def schwefel(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.fabs(x))), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def shubert(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = np.prod(1 * np.cos(2 * x + 1) +
                2 * np.cos(3 * x + 2) +
                3 * np.cos(4 * x + 3) +
                4 * np.cos(5 * x + 4) +
                5 * np.cos(6 * x + 5), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def sine_envelope(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x_curr = x[:-1, :]
    x_next = x[1:, :]
    y = - np.sum((np.sin(np.sqrt(x_next ** 2 + x_curr ** 2) - .5) ** 2)
                 / (.001 * (x_next ** 2 + x_curr ** 2) + 1) ** 2 + .5, axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def sphere(x, df=None):
    x = np.asfarray(np.atleast_1d(x))
    if df is not None:
        x = df(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = np.sum(x ** 2, axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def stochastic(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    i = np.empty(x.shape)
    rng = np.random.default_rng()
    epsilon = np.empty(x.shape)
    for j in range(len(x)):
        i[j] = j + 1
        for k in range(x.shape[1]):
            epsilon[j, k] = rng.uniform(size=1)
    del rng
    y = np.sum(epsilon * np.absolute(x - 1 / i), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def vincent(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = np.sum(np.sin(10 * np.log(x)), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def whitley(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            y += (100 * (x[i] ** 2 - x[j]) ** 2 + (1 - x[j]) ** 2) ** 2 / 4000 \
                 - np.cos(100 * (x[i] ** 2 - x[j]) ** 2 + (1 - x[j]) ** 2) + 1
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


# Xin-She Yang Function 2 (Yang, 2010a,b) (discontinuous, non-differentiable, non-separable, scalable, multimodal)
def xinsheyang02(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = np.sum(np.fabs(x), axis=0) * np.exp(-np.sum(np.sin(x ** 2), axis=0))
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def xinsheyang03(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    beta, m = 15, 3
    y = np.e ** (-np.sum((x / beta) ** (2 * m), axis=0)) \
        - 2 * np.e ** (-np.sum(x ** 2, axis=0)) \
        * np.prod(np.cos(x) ** 2, axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y
