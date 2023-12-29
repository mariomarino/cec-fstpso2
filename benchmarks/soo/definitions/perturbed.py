import math
import numpy as np
from functools import partial


shift_v = 1e6
shrink_v = 1e-20


def ackley_shifted_ext(x, shift, a=20, b=0.2, c=2 * np.pi):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        # gestisce mono-dimensionali per non farli considerare D punti a una dimensione
        x = x[:, np.newaxis]
    shift = np.full_like(x, shift)
    # At this point, x is a matrix where each column is an individual, and each row a variable.
    y = - a * np.exp(-b * np.sqrt(np.mean((x - shift) ** 2, axis=0))) - np.exp(np.mean(np.cos(c * (x - shift)), axis=0)) + a + np.e
    # The axis=0 operation parallelizes the sum of the matrix directly using an efficient NumPy operation.
    if y.ndim == 1:
        # se l'input è uno scalare, ritorno uno scalare
        y = np.squeeze(y)
    return y


def ackley_shrinked_ext(x, shrink, a=20, b=0.2, c=2 * math.pi):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        # gestisce mono-dimensionali per non farli considerare D punti a una dimensione
        x = x[:, np.newaxis]
    # At this point, x is a matrix where each column is an individual, and each row a variable.
    y = - a * np.exp(-b * np.sqrt(np.mean((x * shrink) ** 2, axis=0))) - np.exp(np.mean(np.cos(c * (x * shrink)), axis=0)) + a + np.e
    # The axis=0 operation parallelizes the sum of the matrix directly using an efficient NumPy operation.
    if y.ndim == 1:
        # se l'input è uno scalare, ritorno uno scalare
        y = np.squeeze(y)
    return y


def alpine_shifted_ext(x, shift):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    shift = np.full_like(x, shift)
    y = np.sum(np.absolute((x - shift) * np.sin(x - shift) + .1 * (x - shift)), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def alpine_shrinked_ext(x, shrink):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y = np.sum(np.absolute((x * shrink) * np.sin(x * shrink) + .1 * (x * shrink)), axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def rosenbrock_shifted_ext(x, shift):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x_curr = x[:-1, :]
    x_next = x[1:, :]
    shift = np.full_like(x_curr, shift)
    y = np.sum(100 * ((x_next - shift) - (x_curr - shift) ** 2) ** 2 + ((x_curr - shift) - 1) ** 2, axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def rosenbrock_shrinked_ext(x, shrink):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x_curr = x[:-1, :]
    x_next = x[1:, :]
    y = np.sum(100 * ((x_next * shrink) - (x_curr * shrink) ** 2) ** 2 + ((x_curr * shrink) - 1) ** 2, axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def sphere_shifted_ext(x, shift):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    shift = np.full_like(x, shift)
    y = np.sum((x - shift) ** 2, axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


def sphere_shrinked_ext(x, shrink):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    shrink = 1./shrink
    y = np.sum((x * shrink) ** 2, axis=0)
    if y.ndim == 1:
        y = np.squeeze(y)
    return y


# N-D Perturbed Test Functions A
ackley_shifted = partial(ackley_shifted_ext, shift=shift_v, a=20, b=0.2, c=2 * math.pi)
ackley_shrinked = partial(ackley_shrinked_ext, shrink=shrink_v, a=20, b=0.2, c=2 * math.pi)
alpine01_shifted = partial(alpine_shifted_ext, shift=shift_v)
alpine01_shrinked = partial(alpine_shrinked_ext, shrink=shrink_v)
# N-D Perturbed Test Functions R
rosenbrock_shifted = partial(rosenbrock_shifted_ext, shift=shift_v)
rosenbrock_shrinked = partial(rosenbrock_shrinked_ext, shrink=shrink_v)
# N-D Perturbed Test Functions S
sphere_shifted = partial(sphere_shifted_ext, shift=shift_v)
sphere_shrinked = partial(sphere_shrinked_ext, shrink=shrink_v)
