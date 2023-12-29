import numpy as np


def f(alpha, x) -> np.ndarray:
    assert alpha > 0
    alpha, x = float(alpha), np.asfarray(x)
    return alpha * x / ((alpha - 1) * x + 1)


def inv_f(alpha, x) -> np.ndarray:
    assert alpha > 0
    alpha, x = float(alpha), np.asfarray(x)
    return f(1 / alpha, x)


def g(beta, x) -> np.ndarray:
    assert (beta > 0)
    beta, x = float(beta), np.asfarray(x)
    return np.log(beta * x + 1) / np.log(beta + 1)


def inv_g(beta, x) -> np.ndarray:
    assert (beta > 0)
    beta, x = float(beta), np.asfarray(x)
    return 1 / beta * ((beta + 1) ** x - 1)


def h(gamma, x) -> np.ndarray:
    assert gamma > 0
    gamma, x = float(gamma), np.asfarray(x)
    return x ** gamma


def inv_h(gamma, x) -> np.ndarray:
    assert gamma > 0
    gamma, x = float(gamma), np.asfarray(x)
    return h(1 / gamma, x)


def k(r, q, inv_q, p, x) -> np.ndarray:
    r, p, x = float(r), float(p), np.asfarray(x)
    q_p_arg = np.minimum(np.ones(x.shape), np.maximum(np.zeros(x.shape), x / r))
    inv_q_p_arg = np.minimum(np.ones(x.shape), np.maximum(np.zeros(x.shape), ((x - r) / (1 - r))))
    return np.where(x <= r, r * q(p, q_p_arg), (1 - r) * inv_q(p, inv_q_p_arg) + r)


# Local Bubble Dilation Function
def db(df, r, c, x) -> np.ndarray:
    assert(r > 0)
    x = np.atleast_1d(x)
    c = np.atleast_1d(np.squeeze(c))
    assert(1 <= x.ndim <= 3)
    assert(c.ndim == 1)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim == 3:
        c = c[:, np.newaxis, np.newaxis]
    else:
        c = c[:, np.newaxis]
    assert(len(x) == len(c))
    diff = x - c
    distance = np.linalg.norm(diff, ord=2, axis=0)
    distance[distance == 0] = r
    df_arg = np.minimum(np.ones(distance.shape), np.maximum(np.zeros(distance.shape), distance / r))
    df_out = np.minimum(np.ones(distance.shape), np.maximum(np.zeros(distance.shape), df(df_arg)))
    return np.where((distance > 0) == (distance < r), df_out * diff / distance * r + c, x)


# Pseudo Pacman Effect
def ppe(df, ss, x) -> np.ndarray:
    dx = df(x)
    dimensions = len(dx)
    lwb, upb = list(zip(*ss))
    broadcast_shape = tuple((dimensions, *((1, ) * (dx.ndim - 1))))
    lwb, upb = np.array(lwb).reshape(broadcast_shape), np.array(upb).reshape(broadcast_shape)
    intervals = np.subtract(upb, lwb)
    while np.any(dx < lwb):
        np.add(dx, intervals, out=dx, where=dx < lwb)
    while np.any(dx > upb):
        np.subtract(dx, intervals, out=dx, where=dx > upb)
    return dx


# ----------- CEC 2021 -----------
import math


def identity(param, x):
    return x


def f_(alpha, x):
    n = alpha * x
    d = (alpha - 1) * x + 1
    return n / d


# f inverse is f with 1/alpha
def f_inverse(alpha, x):
    return f(1 / alpha, x)


def g_(beta, x):
    n = math.log(beta * x + 1)
    d = math.log(beta + 1)
    return n / d


def g_inverse(beta, x):
    return 1 / beta * ((beta + 1) ** x - 1)


def h_(gamma, x):
    return x ** gamma


# h inverse is h with 1/gamma
def h_inverse(gamma, x):
    return h(1 / gamma, x)


def folding_operator(r, q, q_inv, p, x):
    # r is the point of attraction/repulsion
    # q is any dilation function
    # q_inv is the inverse of q, it receives p as parameter
    # p is the parameter of the dilation function
    if x <= r:
        return r * q(p, x / r)
    else:
        arg = (x / (1 - r)) - (r / (1 - r))
        return (1 - r) * q_inv(p, arg) + r