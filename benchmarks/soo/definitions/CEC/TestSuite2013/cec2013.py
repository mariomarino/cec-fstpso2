from numpy import ndarray, array, atleast_1d, newaxis, zeros, float64, squeeze
from functools import partial
from os import chdir, getcwd
from typing import Union

from .cec13_test_func import cec13_test_func


def cec13(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2013/')
    cec13_test_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


cec13_f1: partial = partial(cec13, 1)
cec13_f2: partial = partial(cec13, 2)
cec13_f3: partial = partial(cec13, 3)
cec13_f4: partial = partial(cec13, 4)
cec13_f5: partial = partial(cec13, 5)
cec13_f6: partial = partial(cec13, 6)
cec13_f7: partial = partial(cec13, 7)
cec13_f8: partial = partial(cec13, 8)
cec13_f9: partial = partial(cec13, 9)
cec13_f10: partial = partial(cec13, 10)
cec13_f11: partial = partial(cec13, 11)
cec13_f12: partial = partial(cec13, 12)
cec13_f13: partial = partial(cec13, 13)
cec13_f14: partial = partial(cec13, 14)
cec13_f15: partial = partial(cec13, 15)
cec13_f16: partial = partial(cec13, 16)
cec13_f17: partial = partial(cec13, 17)
cec13_f18: partial = partial(cec13, 18)
cec13_f19: partial = partial(cec13, 19)
cec13_f20: partial = partial(cec13, 20)
cec13_f21: partial = partial(cec13, 21)
cec13_f22: partial = partial(cec13, 22)
cec13_f23: partial = partial(cec13, 23)
cec13_f24: partial = partial(cec13, 24)
cec13_f25: partial = partial(cec13, 25)
cec13_f26: partial = partial(cec13, 26)
cec13_f27: partial = partial(cec13, 27)
cec13_f28: partial = partial(cec13, 28)
