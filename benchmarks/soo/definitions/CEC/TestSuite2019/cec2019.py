from numpy import ndarray, array, atleast_1d, newaxis, zeros, float64, squeeze
from functools import partial
from os import chdir, getcwd
from typing import Union

from .cec19_test_func import cec19_test_func


def cec19(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2019/')
    cec19_test_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


cec19_f1: partial = partial(cec19, 1)
cec19_f2: partial = partial(cec19, 2)
cec19_f3: partial = partial(cec19, 3)
cec19_f4: partial = partial(cec19, 4)
cec19_f5: partial = partial(cec19, 5)
cec19_f6: partial = partial(cec19, 6)
cec19_f7: partial = partial(cec19, 7)
cec19_f8: partial = partial(cec19, 8)
cec19_f9: partial = partial(cec19, 9)
cec19_f10: partial = partial(cec19, 10)
