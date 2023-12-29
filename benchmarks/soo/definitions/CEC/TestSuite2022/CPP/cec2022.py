from numpy import ndarray, array, atleast_1d, newaxis, zeros, float64, squeeze
from functools import partial
from os import chdir, getcwd
from typing import Union

from .cec22_test_func import cec22_test_func


def cec22(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2022/CPP/')
    cec22_test_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


cec22_f1: partial = partial(cec22, 1)
cec22_f2: partial = partial(cec22, 2)
cec22_f3: partial = partial(cec22, 3)
cec22_f4: partial = partial(cec22, 4)
cec22_f5: partial = partial(cec22, 5)
cec22_f6: partial = partial(cec22, 6)
cec22_f7: partial = partial(cec22, 7)
cec22_f8: partial = partial(cec22, 8)
cec22_f9: partial = partial(cec22, 9)
cec22_f10: partial = partial(cec22, 10)
cec22_f11: partial = partial(cec22, 11)
cec22_f12: partial = partial(cec22, 12)
