import numpy as np


def cec20_mmf1(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x1, x2 = x
    y1 = np.absolute(x1 - 2)
    y2 = 1 - np.sqrt(np.absolute(x1 - 2)) + 2 * (x2 - np.sin(6 * np.pi * np.absolute(x1 - 2) + np.pi)) ** 2
    if y1.ndim == 1:
        y1 = np.squeeze(y1)
    if y2.ndim == 1:
        y2 = np.squeeze(y2)
    return y1, y2


def cec20_mmf2(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x1, x2 = x
    y1 = x1.copy()
    y2 = np.where((x2 >= 0) == (x2 <= 1),
                  1 - np.sqrt(x1) + 2 * (4 * (x2 - np.sqrt(x1)) ** 2 - 2 * np.cos(
                      (20 * (x2 - np.sqrt(x1)) * np.pi) / np.sqrt(2)) + 2),
                  1 - np.sqrt(x1) + 2 * (4 * (x2 - 1 - np.sqrt(x1)) ** 2 - np.cos(
                      (20 * (x2 - 1 - np.sqrt(x1)) * np.pi) / np.sqrt(2)) + 2))
    if y1.ndim == 1:
        y1 = np.squeeze(y1)
    if y2.ndim == 1:
        y2 = np.squeeze(y2)
    return y1, y2


def cec20_mmf4(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x1, x2 = x
    y1 = np.absolute(x1)
    y2 = np.where((x2 >= 0) == (x2 <= 1),
                  1 - x1 ** 2 + 2 * (x2 - np.sin(np.pi * np.absolute(x1))) ** 2,
                  1 - x1 ** 2 + 2 * (x2 - 1 - np.sin(np.pi * np.absolute(x1))) ** 2)
    if y1.ndim == 1:
        y1 = np.squeeze(y1)
    if y2.ndim == 1:
        y2 = np.squeeze(y2)
    return y1, y2


def cec20_mmf5(x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x1, x2 = x
    y1 = np.absolute(x1 - 2)
    t = 1 - np.sqrt(np.absolute(x1 - 2))
    y2 = np.where((x2 >= -1) == (x2 <= 1),
                  t + 2 * (x2 - np.sin(6 * np.pi * np.absolute(x1 - 2) + np.pi)) ** 2,
                  t + 2 * (x2 - 2 - np.sin(6 * np.pi * np.absolute(x1 - 2) + np.pi)) ** 2)
    if y1.ndim == 1:
        y1 = np.squeeze(y1)
    if y2.ndim == 1:
        y2 = np.squeeze(y2)
    return y1, y2
