import sys
import copy
import functools
import pathlib
import argparse
import math
import pickle

sys.path.insert(0, "code")

from benchmarks.soo.limits import limits
from benchmarks.soo.functions import functions
import numpy as np

from fstpso_original import FuzzyPSO
from pso_ring import PSO_ring
from pso_ring_dilation_5 import PSO_ring_dilation_5
from pso_ring_dilation_6 import PSO_ring_dilation_6
from fstpso_stu import StuFuzzyPSO

method_map = {
    "pso_ring": PSO_ring,
    "fstpso": FuzzyPSO,
    "pso_ring_dilation_5": PSO_ring_dilation_5,
    "pso_ring_dilation_6": PSO_ring_dilation_6,
    "stufstpso": StuFuzzyPSO,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", '-F', dest='fitness', type=int, default="Rastrigin")
    parser.add_argument("--dimensions", '-D', dest='D', type=int)
    parser.add_argument("--run", '-R', dest='R', type=int)
    parser.add_argument("--remedy", '-RN', dest='remedy_name', type=str, default="stufstpso")

    args = parser.parse_args()

    remedy_name = args.remedy_name
    fitness = f"CEC17-F{args.fitness}"

    dir_results_base = f'results_CEC'

    budget = 1e4 * args.D
    budget_str = "4B"

    pathlib.Path(f'{dir_results_base}/{fitness}/{budget_str}').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{dir_results_base}/{fitness}/{budget_str}/gbest_{remedy_name}').mkdir(parents=True,
                                                                                                   exist_ok=True)

    search_space = [limits[fitness]] * args.D
    f = functions[fitness]

    global_best_per_iter = []

    def callback(s):
        global_best_per_iter.append(s.G.CalculatedFitness)

    initial_population = []
    with open(
            f'results_CEC/{fitness}/{budget_str}/populations/{fitness}_{args.D}D_{args.R}R_population.pickle',
            'rb') as f_ip:
        initial_population = pickle.load(f_ip)

    numberofparticles = len(initial_population)
    fp_init = method_map[args.remedy_name]
    FP = fp_init()
    FP.Boundaries = search_space
    FP.FITNESS = f

    est_iterations = int((budget - numberofparticles) / numberofparticles) - 1
    if "fst" in remedy_name:
        FP.set_fitness(f)
        FP.set_search_space(search_space)
        FP.set_swarm_size(numberofparticles)
        FP.NumberOfParticles = numberofparticles
    else:
        FP.set_number_of_particles(numberofparticles)
    FP.Dimensions = args.D
    FP.MaxIterations = est_iterations

    callback_partial = functools.partial(callback)
    callback_param = {
        'interval': 1,
        'function': callback_partial
    }

    if "fst" in remedy_name:
        FP.solve_with_fstpso(callback=callback_param, initial_guess_list=initial_population, max_FEs=budget, max_iter=est_iterations)
    else:
        FP.Solve(None, callback=callback_param, initial_guess_list=initial_population)

    gbest_0 = np.min([f(s) for s in initial_population])
    # gbest initial population
    global_best_per_iter.insert(0, gbest_0)
    gbest_remedy = copy.deepcopy(global_best_per_iter)

    print(f"best, {remedy_name}: {gbest_remedy[-1]}")
    print(f"lens, {remedy_name}: {len(gbest_remedy)}")

    # dump gbest per iter
    with open(
            f'{dir_results_base}/{fitness}/{budget_str}/gbest_{remedy_name}/{fitness}_{args.D}D_{args.R}R_gbest_{remedy_name}.txt',
            "w") as f:
        for i in gbest_remedy:
            f.write(str(i) + "\n")

