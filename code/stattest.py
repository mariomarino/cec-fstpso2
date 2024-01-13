import sys
import copy
import pathlib
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, ".")

ids = [i for i in range(1, 31) if i != 2]
methods = ["FSTPSO", "RINGPSO", "stufstpso", "pso_ring_dilation_5", "pso_ring_dilation_6"]
dim = 30
budget = "4B"
runs = 30

dir_results = "results_CEC"
pathlib.Path(f'{dir_results}/stattest/{dim}D').mkdir(parents=True, exist_ok=True)

arr_best = []

fun_obj_best = {}
arr_tests = []


for fun_id in ids:
    obj_best = {}
    f_name = f"CEC17-F{fun_id}"
    for method in methods:
        best_method = []
        for run in range(runs):
            with open(
                    f"{dir_results}/{f_name}/{budget}/gbest_{method}/{f_name}_{dim}D_{run}R_gbest_{method}.txt",
                    "r") as f:
                best = float(f.readlines()[-2])
                best_method.append(best)
        obj_best[method] = best_method
    fun_obj_best[fun_id] = copy.deepcopy(obj_best)

tests_per_method = {}

for method in methods:
    tests_method = []
    for fun_id in ids:
        obj_tests = {}
        bests_baseline = fun_obj_best[fun_id]['RINGPSO']
        bests_method = fun_obj_best[fun_id][method]
        avg_baseline = np.mean(bests_baseline)
        avg_method = np.mean(bests_method)
        std_baseline = np.std(bests_baseline)
        std_method = np.std(bests_method)
        obj_tests['f'] = fun_id
        obj_tests['mean_0'] = round(avg_baseline, 2)
        obj_tests['mean_1'] = round(avg_method, 2)
        obj_tests['std_0'] = round(std_baseline, 2)
        obj_tests['std_1'] = round(std_method, 2)
        m_error_1 = np.mean(np.array(bests_method) - np.array([fun_id*100] * runs))
        m_error_0 = np.mean(np.array(bests_baseline) - np.array([fun_id * 100] * runs))
        perc_diff = 100*(m_error_0 - m_error_1) / max(m_error_1, m_error_0)
        obj_tests['diff'] = round(perc_diff, 2)
        u, p = mannwhitneyu(bests_method, bests_baseline, alternative="less", method="exact")
        obj_tests['mwu'] = round(u, 2)
        obj_tests['p'] = round(p, 2)
        tests_method.append(copy.deepcopy(obj_tests))
    tests_per_method[method] = copy.deepcopy(tests_method)

    df_method = pd.DataFrame(tests_method)
    df_method.to_csv(f'{dir_results}/stattest/{dim}D/{method}_less.csv')

