import numpy as np

def cec20_mmf1_pf(num):
    f1 = np.linspace(0, 1, num)
    f2 = 1 - np.sqrt(f1)
    return np.stack([f1, f2], axis=0)

def cec20_mmf2_pf(num):
    f1 = np.linspace(0, 1, num)
    f2 = 1 - np.sqrt(f1)
    return np.stack([f1, f2], axis=0)

def cec20_mmf4_pf(num):
    f1 = np.linspace(0, 1, num)
    f2 = 1 - f1 ** 2
    return np.stack([f1, f2], axis=0)

def cec20_mmf5_pf(num):
    f1 = np.linspace(0, 1, num)
    f2 = 1 - np.sqrt(f1)
    return np.stack([f1, f2], axis=0)