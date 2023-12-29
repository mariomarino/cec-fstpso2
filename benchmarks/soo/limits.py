from math import pi
from .definitions.perturbed import shift_v, shrink_v


limits = {
    # ---------------
    # --- STANDARD --
    # ---------------

    # N-D Test Functions A
    'Ackley': [-32, 32],
    'Adjiman': [[-1, 2], [-1, 1]],
    'Alpine01': [-10, 10],
    'Alpine02': [0, 10],
    # N-D Test Functions B
    'Bohachevsky': [-15, 15],
    'Bukin06': [[-15, -5], [-3, 3]],
    # N-D Test Functions C
    'Cross_in_tray': [-10, 10],
    # N-D Test Functions D
    'Damavandi': [0, 14],
    'Deceptive': [0, 1],
    'DeVilliersGlasser02': [1, 60],
    'Drop_wave': [-5.12, 5.12],
    # N-D Test Functions E
    'EggHolder': [-512, 512],
    # N-D Test Functions F
    'Ferretti': [-10, 10],
    # N-D Test Functions G
    'Griewank': [-600, 600],
    # N-D Test Functions H
    'Holder_table': [-10, 10],
    # N-D Test Functions M
    'Michalewicz': [0, pi],
    'Mishra01': [0, 1],
    # N-D Test Functions N
    'Nobile1': [1e-10, 10],
    'Nobile2': [1e-10, 10],
    'Nobile3': [1e-10, 10],
    # N-D Test Functions P
    'Plateau': [-5.12, 5.12],
    # N-D Test Functions Q
    'Quintic': [-10, 10],
    # N-D Test Functions R
    'Rastrigin': [-5, 5],
    'Ripple01': [0, 1],
    'Rosenbrock': [-5, 10],
    # N-D Test Functions S
    'Salomon': [-100, 100],
    'Schwefel': [-512, 512],
    'Shubert': [-10, 10],
    'Sphere': [-5.12, 5.12],
    'Stochastic': [-5, 5],
    # N-D Test Functions V
    'Vincent': [0.25, 10],
    # N-D Test Functions W
    'Whitley': [-10.24, 10.24],
    # N-D Test Functions X
    'XinSheYang02': [-2 * pi, 2 * pi],
    'XinSheYang03': [-20, 20],

    # ----------------
    # --- PERTURBED --
    # ----------------

    # N-D Perturbed Test Functions A
    'Ackley_shifted': [-30 + shift_v, 30 + shift_v],
    'Ackley_shrinked': [-shrink_v, shrink_v],
    'Alpine01_shifted': [-10 + shift_v, 10 + shift_v],
    'Alpine01_shrinked': [-shrink_v, shrink_v],
    # N-D Perturbed Test Functions R
    'Rosenbrock_shifted': [-5 + shift_v, 10 + shift_v],
    'Rosenbrock_shrinked': [-shrink_v, shrink_v],
    # N-D Perturbed Test Functions S
    'Sphere_shifted': [-5.12 + shift_v, 5.12 + shift_v],
    'Sphere_shrinked': [-shrink_v, shrink_v]
}

# CEC 2022
limits.update({'CEC22-F' + str(i): [-100, 100] for i in range(1, 13)})

# CEC 2021
limits.update({'CEC21-F' + str(i) + '-Basic': [-100, 100] for i in range(1, 11)})
limits.update({'CEC21-F' + str(i) + '-Bias': [-100, 100] for i in range(1, 11)})
limits.update({'CEC21-F' + str(i) + '-Bias&Rotation': [-100, 100] for i in range(1, 11)})
limits.update({'CEC21-F' + str(i) + '-Bias&Shift': [-100, 100] for i in range(1, 11)})
limits.update({'CEC21-F' + str(i) + '-Bias&Shift&Rotation': [-100, 100] for i in range(1, 11)})
limits.update({'CEC21-F' + str(i) + '-Rotation': [-100, 100] for i in range(1, 11)})
limits.update({'CEC21-F' + str(i) + '-Shift': [-100, 100] for i in range(1, 11)})
limits.update({'CEC21-F' + str(i) + '-Shift&Rotation': [-100, 100] for i in range(1, 11)})

# CEC 2019
limits.update({'CEC19-F1': [-8192, 8192]})
limits.update({'CEC19-F2': [-16384, 16384]})
limits.update({'CEC19-F3': [-4, 4]})
limits.update({'CEC19-F' + str(i): [-100, 100] for i in range(4, 11)})

# CEC 2017
limits.update({'CEC17-F' + str(i): [-100, 100] for i in range(1, 31)})

# CEC 2014
limits.update({'CEC14-F' + str(i): [-100, 100] for i in range(1, 31)})

# CEC 2013
limits.update({'CEC13-F' + str(i): [-100, 100] for i in range(1, 29)})

# CEC 2005
limits.update({'CEC05-F' + str(i): [-100, 100] for i in [1, 2, 3, 4, 5, 6, 7, 14, 25]})
limits.update({'CEC05-F' + str(i): [-5, 5] for i in [9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]})
limits.update({'CEC05-F8': [-32, 32]})
limits.update({'CEC05-F11': [-0.5, 0.5]})
limits.update({'CEC05-F12': [-pi, pi]})
limits.update({'CEC05-F13': [-3, 1]})
