from numpy import ndarray, array, atleast_1d, newaxis, zeros, float64, squeeze
from functools import partial
from os import chdir, getcwd
from typing import Union

from .cec21_test_func import cec21_basic_func, cec21_bias_func, cec21_bias_rot_func, cec21_bias_shift_func,\
    cec21_bias_shift_rot_func, cec21_rot_func, cec21_shift_func, cec21_shift_rot_func


def cec21_basic(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2021/')
    cec21_basic_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


def cec21_bias(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2021/')
    cec21_bias_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


def cec21_bias_rot(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2021/')
    cec21_bias_rot_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


def cec21_bias_shift(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2021/')
    cec21_bias_shift_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


def cec21_bias_shift_rot(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2021/')
    cec21_bias_shift_rot_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


def cec21_rot(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2021/')
    cec21_rot_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


def cec21_shift(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2021/')
    cec21_shift_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


def cec21_shift_rot(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./benchmarks/soo/definitions/CEC/TestSuite2021/')
    cec21_shift_rot_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


# CEC 2021 - Basic
cec21_f1_basic: partial = partial(cec21_basic, 1)
cec21_f2_basic: partial = partial(cec21_basic, 2)
cec21_f3_basic: partial = partial(cec21_basic, 3)
cec21_f4_basic: partial = partial(cec21_basic, 4)
cec21_f5_basic: partial = partial(cec21_basic, 5)
cec21_f6_basic: partial = partial(cec21_basic, 6)
cec21_f7_basic: partial = partial(cec21_basic, 7)
cec21_f8_basic: partial = partial(cec21_basic, 8)
cec21_f9_basic: partial = partial(cec21_basic, 9)
cec21_f10_basic: partial = partial(cec21_basic, 10)
# CEC 2021 - Bias
cec21_f1_bias: partial = partial(cec21_bias, 1)
cec21_f2_bias: partial = partial(cec21_bias, 2)
cec21_f3_bias: partial = partial(cec21_bias, 3)
cec21_f4_bias: partial = partial(cec21_bias, 4)
cec21_f5_bias: partial = partial(cec21_bias, 5)
cec21_f6_bias: partial = partial(cec21_bias, 6)
cec21_f7_bias: partial = partial(cec21_bias, 7)
cec21_f8_bias: partial = partial(cec21_bias, 8)
cec21_f9_bias: partial = partial(cec21_bias, 9)
cec21_f10_bias: partial = partial(cec21_bias, 10)
# CEC 2021 - Bias & Rotation
cec21_f1_bias_rot: partial = partial(cec21_bias_rot, 1)
cec21_f2_bias_rot: partial = partial(cec21_bias_rot, 2)
cec21_f3_bias_rot: partial = partial(cec21_bias_rot, 3)
cec21_f4_bias_rot: partial = partial(cec21_bias_rot, 4)
cec21_f5_bias_rot: partial = partial(cec21_bias_rot, 5)
cec21_f6_bias_rot: partial = partial(cec21_bias_rot, 6)
cec21_f7_bias_rot: partial = partial(cec21_bias_rot, 7)
cec21_f8_bias_rot: partial = partial(cec21_bias_rot, 8)
cec21_f9_bias_rot: partial = partial(cec21_bias_rot, 9)
cec21_f10_bias_rot: partial = partial(cec21_bias_rot, 10)
# CEC 2021 - Bias & Shift
cec21_f1_bias_shift: partial = partial(cec21_bias_shift, 1)
cec21_f2_bias_shift: partial = partial(cec21_bias_shift, 2)
cec21_f3_bias_shift: partial = partial(cec21_bias_shift, 3)
cec21_f4_bias_shift: partial = partial(cec21_bias_shift, 4)
cec21_f5_bias_shift: partial = partial(cec21_bias_shift, 5)
cec21_f6_bias_shift: partial = partial(cec21_bias_shift, 6)
cec21_f7_bias_shift: partial = partial(cec21_bias_shift, 7)
cec21_f8_bias_shift: partial = partial(cec21_bias_shift, 8)
cec21_f9_bias_shift: partial = partial(cec21_bias_shift, 9)
cec21_f10_bias_shift: partial = partial(cec21_bias_shift, 10)
# CEC 2021 - Bias & Shift & Rotation
cec21_f1_bias_shift_rot: partial = partial(cec21_bias_shift_rot, 1)
cec21_f2_bias_shift_rot: partial = partial(cec21_bias_shift_rot, 2)
cec21_f3_bias_shift_rot: partial = partial(cec21_bias_shift_rot, 3)
cec21_f4_bias_shift_rot: partial = partial(cec21_bias_shift_rot, 4)
cec21_f5_bias_shift_rot: partial = partial(cec21_bias_shift_rot, 5)
cec21_f6_bias_shift_rot: partial = partial(cec21_bias_shift_rot, 6)
cec21_f7_bias_shift_rot: partial = partial(cec21_bias_shift_rot, 7)
cec21_f8_bias_shift_rot: partial = partial(cec21_bias_shift_rot, 8)
cec21_f9_bias_shift_rot: partial = partial(cec21_bias_shift_rot, 9)
cec21_f10_bias_shift_rot: partial = partial(cec21_bias_shift_rot, 10)
# CEC 2021 - Rotation
cec21_f1_rot: partial = partial(cec21_rot, 1)
cec21_f2_rot: partial = partial(cec21_rot, 2)
cec21_f3_rot: partial = partial(cec21_rot, 3)
cec21_f4_rot: partial = partial(cec21_rot, 4)
cec21_f5_rot: partial = partial(cec21_rot, 5)
cec21_f6_rot: partial = partial(cec21_rot, 6)
cec21_f7_rot: partial = partial(cec21_rot, 7)
cec21_f8_rot: partial = partial(cec21_rot, 8)
cec21_f9_rot: partial = partial(cec21_rot, 9)
cec21_f10_rot: partial = partial(cec21_rot, 10)
# CEC 2021 - Shift
cec21_f1_shift: partial = partial(cec21_shift, 1)
cec21_f2_shift: partial = partial(cec21_shift, 2)
cec21_f3_shift: partial = partial(cec21_shift, 3)
cec21_f4_shift: partial = partial(cec21_shift, 4)
cec21_f5_shift: partial = partial(cec21_shift, 5)
cec21_f6_shift: partial = partial(cec21_shift, 6)
cec21_f7_shift: partial = partial(cec21_shift, 7)
cec21_f8_shift: partial = partial(cec21_shift, 8)
cec21_f9_shift: partial = partial(cec21_shift, 9)
cec21_f10_shift: partial = partial(cec21_shift, 10)
# CEC 2021 - Shift & Rotation
cec21_f1_shift_rot: partial = partial(cec21_shift_rot, 1)
cec21_f2_shift_rot: partial = partial(cec21_shift_rot, 2)
cec21_f3_shift_rot: partial = partial(cec21_shift_rot, 3)
cec21_f4_shift_rot: partial = partial(cec21_shift_rot, 4)
cec21_f5_shift_rot: partial = partial(cec21_shift_rot, 5)
cec21_f6_shift_rot: partial = partial(cec21_shift_rot, 6)
cec21_f7_shift_rot: partial = partial(cec21_shift_rot, 7)
cec21_f8_shift_rot: partial = partial(cec21_shift_rot, 8)
cec21_f9_shift_rot: partial = partial(cec21_shift_rot, 9)
cec21_f10_shift_rot: partial = partial(cec21_shift_rot, 10)
