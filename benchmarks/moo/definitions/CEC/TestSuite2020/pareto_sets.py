import numpy as np

def cec20_mmf1_ps(num):
    x1 = np.linspace(1, 3, num)
    x2 = np.sin(6 * np.pi * np.absolute(x1 - 2) + np.pi)
    return np.stack([x1, x2], axis=0)

def cec20_mmf2_ps(num):
    x2 = np.linspace(0, 2, num)
    x1 = np.where((x2 >= 0) == (x2 <= 1), x2 ** 2, (x2 - 1) ** 2)
    return np.stack([x1, x2], axis=0)

def cec20_mmf4_ps(num):
    x1 = np.linspace(-1, 1, num)
    a = np.sin(np.pi * np.absolute(x1))
    b = a + 1
    x2 = np.where((a >= 0) == (a <= 1), a, np.where((b > 1) == (b <= 2), b, x1))
    return np.stack([x1, x2], axis=0)

def cec20_mmf5_ps(num):
    x1 = np.linspace(-1, 3, num)
    a = np.sin(6 * np.pi * np.absolute(x1 - 2) + np.pi)
    b = a + 2
    x2 = np.where((a >= -1) == (a <= 1), a, np.where((b > 1) == (b <= 3), b, x1))
    return np.stack([x1, x2], axis=0)