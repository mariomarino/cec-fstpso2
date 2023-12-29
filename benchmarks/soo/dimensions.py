dimensions = {
    # ---------------
    # --- STANDARD --
    # ---------------

    # N-D Test Functions A
    'Ackley': tuple(i + 1 for i in range(100)),
    'Adjiman': (2,),
    'Alpine01': tuple(i + 1 for i in range(100)),
    'Alpine02': tuple(i + 1 for i in range(100)),
    # N-D Test Functions B
    'Bohachevsky': tuple(i + 1 for i in range(100)),
    'Bukin06': (2,),
    # N-D Test Functions C
    'Cross_in_tray': (2,),
    # N-D Test Functions D
    'Damavandi': tuple(i + 1 for i in range(100)),
    'Deceptive': tuple(i + 1 for i in range(100)),
    'DeVilliersGlasser02': (5,),
    'Drop_wave': (2,),
    # N-D Test Functions E
    'EggHolder': (2,),
    # N-D Test Functions F
    'Ferretti': tuple(i + 1 for i in range(100)),
    # N-D Test Functions G
    'Griewank': tuple(i + 1 for i in range(100)),
    # N-D Test Functions H
    'Holder_table': (2,),
    # N-D Test Functions M
    'Michalewicz': tuple(i + 1 for i in range(100)),
    'Mishra01': tuple(i + 1 for i in range(100)),
    # N-D Test Functions N
    'Nobile1': tuple(i + 1 for i in range(100)),
    'Nobile2': tuple(i + 1 for i in range(100)),
    'Nobile3': tuple(i + 1 for i in range(100)),
    # N-D Test Functions P
    'Plateau': tuple(i + 1 for i in range(100)),
    # N-D Test Functions Q
    'Quintic': tuple(i + 1 for i in range(100)),
    # N-D Test Functions R
    'Rastrigin': tuple(i + 1 for i in range(100)),
    'Ripple01': tuple(i + 1 for i in range(100)),
    'Rosenbrock': tuple(i + 1 for i in range(100)),
    # N-D Test Functions S
    'Salomon': tuple(i + 1 for i in range(100)),
    'Schwefel': tuple(i + 1 for i in range(100)),
    'Shubert': tuple(i + 1 for i in range(100)),
    'Sphere': (2,),
    'Stochastic': tuple(i + 1 for i in range(100)),
    # N-D Test Functions V
    'Vincent': tuple(i + 1 for i in range(100)),
    # N-D Test Functions W
    'Whitley': tuple(i + 1 for i in range(100)),
    # N-D Test Functions X
    'XinSheYang02': tuple(i + 1 for i in range(100)),
    'XinSheYang03': tuple(i + 1 for i in range(100)),

    # ----------------
    # --- PERTURBED --
    # ----------------

    # N-D Perturbed Test Functions A
    'Ackley_shifted': tuple(i + 1 for i in range(100)),
    'Ackley_shrinked': tuple(i + 1 for i in range(100)),
    'Alpine01_shrinked': tuple(i + 1 for i in range(100)),
    'Alpine01_shifted': tuple(i + 1 for i in range(100)),
    # N-D Perturbed Test Functions R
    'Rosenbrock_shrinked': tuple(i + 1 for i in range(100)),
    'Rosenbrock_shifted': tuple(i + 1 for i in range(100)),
    # N-D Perturbed Test Functions S
    'Sphere_shrinked': tuple(i + 1 for i in range(100)),
    'Sphere_shifted': tuple(i + 1 for i in range(100))
}

# CEC 2022
dimensions.update({'CEC22-F' + str(i): (2, 10, 20) for i in range(1, 6)})
dimensions.update({'CEC22-F' + str(i): (10, 20) for i in range(6, 9)})
dimensions.update({'CEC22-F' + str(i): (2, 10, 20) for i in range(9, 13)})

# CEC 2021
dimensions.update({'CEC21-F' + str(i) + '-Basic': (2, 10, 20) for i in range(1, 5)})
dimensions.update({'CEC21-F' + str(i) + '-Basic': (10, 20) for i in range(5, 8)})
dimensions.update({'CEC21-F' + str(i) + '-Basic': (2, 10, 20) for i in range(8, 11)})
dimensions.update({'CEC21-F' + str(i) + '-Bias': (2, 10, 20) for i in range(1, 5)})
dimensions.update({'CEC21-F' + str(i) + '-Bias': (10, 20) for i in range(5, 8)})
dimensions.update({'CEC21-F' + str(i) + '-Bias': (2, 10, 20) for i in range(8, 11)})
dimensions.update({'CEC21-F' + str(i) + '-Bias&Rotation': (2, 10, 20) for i in range(1, 5)})
dimensions.update({'CEC21-F' + str(i) + '-Bias&Rotation': (10, 20) for i in range(5, 8)})
dimensions.update({'CEC21-F' + str(i) + '-Bias&Rotation': (2, 10, 20) for i in range(8, 11)})
dimensions.update({'CEC21-F' + str(i) + '-Bias&Shift': (2, 10, 20) for i in range(1, 5)})
dimensions.update({'CEC21-F' + str(i) + '-Bias&Shift': (10, 20) for i in range(5, 8)})
dimensions.update({'CEC21-F' + str(i) + '-Bias&Shift': (2, 10, 20) for i in range(8, 11)})
dimensions.update({'CEC21-F' + str(i) + '-Bias&Shift&Rotation': (2, 10, 20) for i in range(1, 5)})
dimensions.update({'CEC21-F' + str(i) + '-Bias&Shift&Rotation': (10, 20) for i in range(5, 8)})
dimensions.update({'CEC21-F' + str(i) + '-Bias&Shift&Rotation': (2, 10, 20) for i in range(8, 11)})
dimensions.update({'CEC21-F' + str(i) + '-Rotation': (2, 10, 20) for i in range(1, 5)})
dimensions.update({'CEC21-F' + str(i) + '-Rotation': (10, 20) for i in range(5, 8)})
dimensions.update({'CEC21-F' + str(i) + '-Rotation': (2, 10, 20) for i in range(8, 11)})
dimensions.update({'CEC21-F' + str(i) + '-Shift': (2, 10, 20) for i in range(1, 5)})
dimensions.update({'CEC21-F' + str(i) + '-Shift': (10, 20) for i in range(5, 8)})
dimensions.update({'CEC21-F' + str(i) + '-Shift': (2, 10, 20) for i in range(8, 11)})
dimensions.update({'CEC21-F' + str(i) + '-Shift&Rotation': (2, 10, 20) for i in range(1, 5)})
dimensions.update({'CEC21-F' + str(i) + '-Shift&Rotation': (10, 20) for i in range(5, 8)})
dimensions.update({'CEC21-F' + str(i) + '-Shift&Rotation': (2, 10, 20) for i in range(8, 11)})

# CEC 2019
dimensions.update({'CEC19-F1': (9,)})
dimensions.update({'CEC19-F2': (16,)})
dimensions.update({'CEC19-F3': (18,)})
dimensions.update({'CEC19-F' + str(i): (10,) for i in range(4, 11)})

# CEC 2017
dimensions.update({'CEC17-F' + str(i): (2, 10, 20, 30, 50, 100) for i in range(1, 11)})
dimensions.update({'CEC17-F' + str(i): (10, 30, 50, 100) for i in range(11, 20)})
dimensions.update({'CEC17-F' + str(i): (10, 20, 30, 50, 100) for i in range(20, 21)})
dimensions.update({'CEC17-F' + str(i): (2, 10, 20, 30, 50, 100) for i in range(21, 29)})
dimensions.update({'CEC17-F' + str(i): (2, 10, 30, 50, 100) for i in range(29, 31)})

# CEC 2014
dimensions.update({'CEC14-F' + str(i): (2, 10, 20, 30, 50, 100) for i in range(1, 17)})
dimensions.update({'CEC14-F' + str(i): (10, 20, 30, 50, 100) for i in range(17, 23)})
dimensions.update({'CEC14-F' + str(i): (2, 10, 20, 30, 50, 100) for i in range(23, 29)})
dimensions.update({'CEC14-F' + str(i): (10, 20, 30, 50, 100) for i in range(29, 31)})

# CEC 2013
dimensions.update({'CEC13-F' + str(i): (2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100) for i in range(1, 29)})

# CEC 2005
dimensions.update({'CEC05-F' + str(i): (50,) for i in [3, 7, 8, 10, 11, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]})
dimensions.update({'CEC05-F' + str(i): (100,) for i in [1, 2, 4, 5, 6, 9, 12, 13, 15]})
