from .definitions.CEC.TestSuite2020.cec2020 import cec20_mmf1, cec20_mmf2, cec20_mmf4, cec20_mmf5
from .definitions.standard import fonseca, kursawe
from .definitions.PyMOO.test_problems import dtlz1, dtlz2, dtlz3, dtlz4, dtlz5, dtlz6, dtlz7, wfg1, wfg2, wfg3, wfg4,\
    wfg5, wfg6, wfg7, wfg8, wfg9


functions = {
    # CEC 2020
    'CEC20-F1': cec20_mmf1,
    'CEC20-F2': cec20_mmf2,
    'CEC20-F3': cec20_mmf4,
    'CEC20-F4': cec20_mmf5,
    # DEAP
    'Fonseca': fonseca,
    'Kursawe': kursawe,
    # PyMOO
    'DTLZ1': dtlz1,
    'DTLZ2': dtlz2,
    'DTLZ3': dtlz3,
    'DTLZ4': dtlz4,
    'DTLZ5': dtlz5,
    'DTLZ6': dtlz6,
    'DTLZ7': dtlz7,
    'WFG1': wfg1,
    'WFG2': wfg2,
    'WFG3': wfg3,
    'WFG4': wfg4,
    'WFG5': wfg5,
    'WFG6': wfg6,
    'WFG7': wfg7,
    'WFG8': wfg8,
    'WFG9': wfg9,
}
