from numpy import ndarray, array, atleast_1d, newaxis, zeros, float64, squeeze
from functools import partial
from os import chdir, getcwd
from typing import Union

from .cec14_test_func import cec14_test_func


def cec14(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2014/')
    cec14_test_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


cec14_f1: partial = partial(cec14, 1)
cec14_f2: partial = partial(cec14, 2)
cec14_f3: partial = partial(cec14, 3)
cec14_f4: partial = partial(cec14, 4)
cec14_f5: partial = partial(cec14, 5)
cec14_f6: partial = partial(cec14, 6)
cec14_f7: partial = partial(cec14, 7)
cec14_f8: partial = partial(cec14, 8)
cec14_f9: partial = partial(cec14, 9)
cec14_f10: partial = partial(cec14, 10)
cec14_f11: partial = partial(cec14, 11)
cec14_f12: partial = partial(cec14, 12)
cec14_f13: partial = partial(cec14, 13)
cec14_f14: partial = partial(cec14, 14)
cec14_f15: partial = partial(cec14, 15)
cec14_f16: partial = partial(cec14, 16)
cec14_f17: partial = partial(cec14, 17)
cec14_f18: partial = partial(cec14, 18)
cec14_f19: partial = partial(cec14, 19)
cec14_f20: partial = partial(cec14, 20)
cec14_f21: partial = partial(cec14, 21)
cec14_f22: partial = partial(cec14, 22)
cec14_f23: partial = partial(cec14, 23)
cec14_f24: partial = partial(cec14, 24)
cec14_f25: partial = partial(cec14, 25)
cec14_f26: partial = partial(cec14, 26)
cec14_f27: partial = partial(cec14, 27)
cec14_f28: partial = partial(cec14, 28)
cec14_f29: partial = partial(cec14, 29)
cec14_f30: partial = partial(cec14, 30)
