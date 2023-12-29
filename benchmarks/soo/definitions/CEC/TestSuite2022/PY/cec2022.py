from os import chdir, getcwd
from numpy import ndarray, asfarray, atleast_1d, newaxis, squeeze, float64
from functools import partial
from typing import Union

from .cec22_test_func import cec2022_func


def cec22(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = asfarray(atleast_1d(x))
    if x.ndim == 1:
        x = x[:, newaxis]

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2022/PY/')
    cec = cec2022_func(func_num=func_id)
    f = cec.values(x).ObjFunc
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
