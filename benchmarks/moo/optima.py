from .definitions.CEC.TestSuite2020.pareto_sets import cec20_mmf1_ps, cec20_mmf2_ps, cec20_mmf4_ps, cec20_mmf5_ps
from .definitions.CEC.TestSuite2020.pareto_fronts import cec20_mmf1_pf, cec20_mmf2_pf, cec20_mmf4_pf, cec20_mmf5_pf
from .definitions.PyMOO.pareto_sets import dtlz1_ps, dtlz2_ps, dtlz3_ps, dtlz4_ps, dtlz5_ps, dtlz6_ps, dtlz7_ps,\
    wfg1_ps, wfg2_ps, wfg3_ps, wfg4_ps, wfg5_ps, wfg6_ps, wfg7_ps, wfg8_ps, wfg9_ps
from .definitions.PyMOO.pareto_fronts import dtlz1_pf, dtlz2_pf, dtlz3_pf, dtlz4_pf, dtlz5_pf, dtlz6_pf, dtlz7_pf,\
    wfg1_pf, wfg2_pf, wfg3_pf, wfg4_pf, wfg5_pf, wfg6_pf, wfg7_pf, wfg8_pf, wfg9_pf


def exception():
    raise Exception('Not implemented yet.')


optima = {
    # CEC 2020
    'CEC20-F1': {'pareto_set': cec20_mmf1_ps, 'pareto_front': cec20_mmf1_pf},
    'CEC20-F2': {'pareto_set': cec20_mmf2_ps, 'pareto_front': cec20_mmf2_pf},
    'CEC20-F3': {'pareto_set': cec20_mmf4_ps, 'pareto_front': cec20_mmf4_pf},
    'CEC20-F4': {'pareto_set': cec20_mmf5_ps, 'pareto_front': cec20_mmf5_pf},
    # STANDARD
    'Fonseca': {'pareto_set': exception, 'pareto_front': exception},
    'Kursawe': {'pareto_set': exception, 'pareto_front': exception},
    # PyMOO
    'DTLZ1': {'pareto_set': dtlz1_ps, 'pareto_front': dtlz1_pf},
    'DTLZ2': {'pareto_set': dtlz2_ps, 'pareto_front': dtlz2_pf},
    'DTLZ3': {'pareto_set': dtlz3_ps, 'pareto_front': dtlz3_pf},
    'DTLZ4': {'pareto_set': dtlz4_ps, 'pareto_front': dtlz4_pf},
    'DTLZ5': {'pareto_set': dtlz5_ps, 'pareto_front': dtlz5_pf},
    'DTLZ6': {'pareto_set': dtlz6_ps, 'pareto_front': dtlz6_pf},
    'DTLZ7': {'pareto_set': dtlz7_ps, 'pareto_front': dtlz7_pf},

    'WFG1': {'pareto_set': wfg1_ps, 'pareto_front': wfg1_pf},
    'WFG2': {'pareto_set': wfg2_ps, 'pareto_front': wfg2_pf},
    'WFG3': {'pareto_set': wfg3_ps, 'pareto_front': wfg3_pf},
    'WFG4': {'pareto_set': wfg4_ps, 'pareto_front': wfg4_pf},
    'WFG5': {'pareto_set': wfg5_ps, 'pareto_front': wfg5_pf},
    'WFG6': {'pareto_set': wfg6_ps, 'pareto_front': wfg6_pf},
    'WFG7': {'pareto_set': wfg7_ps, 'pareto_front': wfg7_pf},
    'WFG8': {'pareto_set': wfg8_ps, 'pareto_front': wfg8_pf},
    'WFG9': {'pareto_set': wfg9_ps, 'pareto_front': wfg9_pf},
}
