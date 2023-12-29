import numpy as np
from pymoo.problems import get_problem


def fonseca(n_obj: int, x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y1 = 1 - np.exp(-np.sum((x - 1 / np.sqrt(3)) ** 2, axis=0))
    y2 = 1 - np.exp(-np.sum((x + 1 / np.sqrt(3)) ** 2, axis=0))
    if y1.ndim == 1:
        y1 = np.squeeze(y1)
    if y2.ndim == 1:
        y2 = np.squeeze(y2)
    return y1, y2


def kursawe(n_obj: int, x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x_i = x[:-1, :]
    x_i_plus_1 = x[1:, :]
    y1 = np.sum(-10 * np.exp(-.2 * np.sqrt(x_i ** 2 + x_i_plus_1 ** 2)), axis=0)
    y2 = np.sum(np.absolute(x) ** .8 + 5 * np.sin(x ** 3), axis=0)
    if y1.ndim == 1:
        y1 = np.squeeze(y1)
    if y2.ndim == 1:
        y2 = np.squeeze(y2)
    return y1, y2


# def dtlz1(x, m=2):
#     x = np.asfarray(np.atleast_1d(x))
#     if x.ndim == 1:
#         x = x[:, np.newaxis]
#     assert(len(x) >= m)
#     x_m = x[m-1:, :]
#     print('x_m', x_m)
#     g = 100 * (len(x_m) + np.sum((x_m - .5) ** 2 - np.cos(20 * np.pi * (x_m - .5)), axis=0))
#     print('g', g)
#     x_c = x[:m-1, :]
#     print('x_c', x_c)
#     y1 = .5 * (1 + g) * np.prod(x_c, axis=0)
#     if y1.ndim == 1:
#         y1 = np.squeeze(y1)
#     print('y1', y1)
#     y = [y1]
#     for j in reversed(range(m-1)):
#         x_j = x[:j, :]
#         print('x_j', x_j)
#         y_j = .5 * (1 + g) * (1 - x[j, :]) * np.prod(x_j, axis=0)
#         print('np.prod(x_j, axis=0)', np.prod(x_j, axis=0))
#         print('x[j, :]', x[j, :])
#         print('(1 - x[j, :])', (1 - x[j, :]))
#         print('y_j', y_j)
#         if y_j.ndim == 1:
#             y_j = np.squeeze(y_j)
#         y.append(y_j)
#         print('y[0].shape', y[0].shape)
#         print('y[0]', y[0])
#         print('y[1].shape', y[1].shape)
#         print('y[1]', y[1])
#         # exit(1)
#     return y


# def dtlz1(x):
#     x = np.asfarray(np.atleast_1d(x))
#     if x.ndim == 1:
#         x = x[:, np.newaxis]
#     problem = get_problem('dtlz1', n_var=2, n_obj=2)
#     y = problem.evaluate(X=x.T).T
#     return tuple(np.squeeze(z) if z.ndim == 1 else z for z in y)
